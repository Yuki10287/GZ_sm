"""Semi-automatic downloader for gasoline/diesel customs trade data.

This script intentionally does not bypass captchas. It opens a visible browser,
fills what it can, pauses for manual captcha/confirmation, and renames each
download to the project convention:

    data/task2/customs_raw/gasoline_import_2016.xlsx

The China Customs statistics page changes its front-end occasionally, so two
flows are supported:

1. full: try to fill fuel, flow, monthly display, all-region options, year.
2. assisted: you manually set the fixed conditions first; the script loops
   years and captures/renames one download per year.

Examples:
    python scripts/download_customs_trade_semiauto.py --mode assisted --fuel gasoline --flow import
    python scripts/download_customs_trade_semiauto.py --mode full
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "task2" / "customs_raw"
DEFAULT_URL = "http://stats.customs.gov.cn/"

TASKS = [
    ("gasoline", "import"),
    ("gasoline", "export"),
    ("diesel", "import"),
    ("diesel", "export"),
]

COMMODITIES = {
    "gasoline": {
        "name": "汽油",
        "codes": ["27101210", "2710121"],
    },
    "diesel": {
        "name": "柴油",
        "codes": ["27101923", "27101926"],
    },
}

FLOW_TEXT = {
    "import": ["进口", "进境"],
    "export": ["出口", "出境"],
}


@dataclass(frozen=True)
class DownloadTask:
    fuel: str
    flow: str
    year: int

    @property
    def output_path(self) -> Path:
        return RAW_DIR / f"{self.fuel}_{self.flow}_{self.year}.xlsx"


def require_playwright():
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "缺少 Playwright。先运行：\n"
            "  pip install playwright\n"
            "  python -m playwright install chromium\n"
        ) from exc
    return sync_playwright, PlaywrightTimeoutError


def normalize_download(download_path: Path, final_path: Path) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    suffix = download_path.suffix.lower()
    if suffix not in {".xlsx", ".xls", ".csv"}:
        suffix = ".xlsx"
    final_path = final_path.with_suffix(suffix)
    if final_path.exists():
        backup = final_path.with_name(final_path.stem + f"_old_{int(time.time())}" + final_path.suffix)
        final_path.replace(backup)
        print(f"已存在旧文件，已备份为：{backup.name}")
    shutil.move(str(download_path), str(final_path))
    print(f"已保存：{final_path}")


class CustomsPageHelper:
    def __init__(self, page, timeout_ms: int = 2500) -> None:
        self.page = page
        self.timeout_ms = timeout_ms

    def _try(self, func, description: str) -> bool:
        try:
            func()
            print(f"  OK: {description}")
            return True
        except Exception as exc:  # noqa: BLE001 - page selectors are best-effort.
            print(f"  跳过: {description} ({type(exc).__name__})")
            return False

    def click_text(self, texts: list[str], exact: bool = False) -> bool:
        for text in texts:
            pattern = re.compile(re.escape(text))

            def _click() -> None:
                self.page.get_by_text(pattern, exact=exact).first.click(timeout=self.timeout_ms)

            if self._try(_click, f"点击文本 {text}"):
                return True
        return False

    def fill_label_or_placeholder(self, labels: list[str], value: str) -> bool:
        for label in labels:
            candidates = [
                lambda label=label: self.page.get_by_label(label).first,
                lambda label=label: self.page.get_by_placeholder(label).first,
                lambda label=label: self.page.locator(
                    f"xpath=//*[contains(normalize-space(.), '{label}')]/following::input[1]"
                ).first,
            ]
            for make_locator in candidates:
                def _fill(make_locator=make_locator) -> None:
                    loc = make_locator()
                    loc.fill(value, timeout=self.timeout_ms)

                if self._try(_fill, f"填写 {label}={value}"):
                    return True
        return False

    def select_label(self, labels: list[str], values: list[str]) -> bool:
        for label in labels:
            for value in values:
                selectors = [
                    lambda label=label: self.page.get_by_label(label).first,
                    lambda label=label: self.page.locator(
                        f"xpath=//*[contains(normalize-space(.), '{label}')]/following::select[1]"
                    ).first,
                ]
                for make_locator in selectors:
                    def _select(make_locator=make_locator, value=value) -> None:
                        make_locator().select_option(label=value, timeout=self.timeout_ms)

                    if self._try(_select, f"选择 {label}={value}"):
                        return True
        return False

    def set_year(self, year: int) -> bool:
        labels = ["年份", "时间", "数据年月", "统计时间", "起始年份", "开始年份"]
        if self.select_label(labels, [str(year), f"{year}年"]):
            return True
        return self.fill_label_or_placeholder(labels, str(year))

    def try_full_conditions(self, fuel: str, flow: str, year: int) -> None:
        commodity = COMMODITIES[fuel]
        print(f"尝试自动设置：{fuel}/{flow}/{year}")
        self.click_text(["查询", "条件查询", "统计查询"])
        self.click_text(["分月展示", "按月展示", "月度", "月份"])
        self.click_text(FLOW_TEXT[flow])
        self.click_text(["全部", "所有"], exact=False)
        filled = False
        for code in commodity["codes"]:
            filled = self.fill_label_or_placeholder(["商品编码", "商品代码", "编码"], code)
            if filled:
                break
        if not filled:
            self.fill_label_or_placeholder(["商品名称", "商品"], commodity["name"])
        self.set_year(year)

    def click_query(self) -> bool:
        return self.click_text(["查询", "搜索", "开始查询", "统计"])

    def click_download(self) -> bool:
        return self.click_text(["下载", "导出", "Excel", "导出Excel", "数据下载"])


def wait_for_one_download(page, helper: CustomsPageHelper, task: DownloadTask, mode: str) -> None:
    print()
    print(f"准备下载 {task.fuel}_{task.flow}_{task.year}")
    print("如果页面出现验证码，请手动完成验证码。")
    print("脚本会先尝试点击查询和下载；若失败，你可以在浏览器里手动点击下载。")
    input("确认页面条件无误后按 Enter 继续...")

    helper.set_year(task.year)
    if mode == "full":
        helper.try_full_conditions(task.fuel, task.flow, task.year)

    helper.click_query()
    input("若出现验证码，请完成验证码并等待结果加载；完成后按 Enter 捕获下载...")

    with page.expect_download(timeout=180_000) as download_info:
        if not helper.click_download():
            print("未能自动点击下载按钮，请在 180 秒内手动点击页面下载按钮。")
    download = download_info.value
    temp_path = Path(download.path())
    normalize_download(temp_path, task.output_path)


def run(args: argparse.Namespace) -> None:
    sync_playwright, _ = require_playwright()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    selected_tasks = TASKS
    if args.fuel != "all" and args.flow != "all":
        selected_tasks = [(args.fuel, args.flow)]
    elif args.fuel != "all":
        selected_tasks = [(args.fuel, flow) for flow in ["import", "export"]]
    elif args.flow != "all":
        selected_tasks = [(fuel, args.flow) for fuel in ["gasoline", "diesel"]]

    with sync_playwright() as p:
        launch_kwargs = {"headless": False, "slow_mo": args.slow_mo}
        if args.browser_channel:
            launch_kwargs["channel"] = args.browser_channel
        browser = p.chromium.launch(**launch_kwargs)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        page.goto(args.url, wait_until="domcontentloaded", timeout=60_000)
        helper = CustomsPageHelper(page, timeout_ms=args.selector_timeout)

        print("浏览器已打开海关统计数据查询平台。")
        print("验证码必须由你手动完成，脚本不会尝试识别或绕过验证码。")

        for fuel, flow in selected_tasks:
            print()
            print("=" * 72)
            print(f"当前任务：{fuel}_{flow}")
            print("商品建议：", COMMODITIES[fuel])
            print("固定条件：分月展示；贸易伙伴/贸易方式/收发货人注册地均选择全部。")
            if args.mode == "assisted":
                input(
                    "请在浏览器中手动设置好该品种和进出口方向的固定条件，"
                    "保留年份可改；设置完成后按 Enter..."
                )

            for year in range(args.start_year, args.end_year + 1):
                task = DownloadTask(fuel=fuel, flow=flow, year=year)
                if task.output_path.exists() and not args.overwrite:
                    print(f"已存在，跳过：{task.output_path.name}")
                    continue
                wait_for_one_download(page, helper, task, args.mode)

        print("全部下载任务结束。")
        if not args.keep_open:
            context.close()
            browser.close()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["assisted", "full"], default="assisted")
    parser.add_argument("--fuel", choices=["all", "gasoline", "diesel"], default="all")
    parser.add_argument("--flow", choices=["all", "import", "export"], default="all")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--selector-timeout", type=int, default=2500)
    parser.add_argument("--slow-mo", type=int, default=100)
    parser.add_argument(
        "--browser-channel",
        choices=["msedge", "chrome", "chrome-beta", "chrome-dev"],
        default="msedge",
        help="Use an installed system browser. This avoids downloading Playwright Chromium.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args(sys.argv[1:]))
