from playwright.async_api import async_playwright, Browser, Page, Playwright, BrowserContext


class BrowserController:
    """Singleton Playwright browser instance shared across all graph nodes."""

    _playwright: Playwright | None = None
    _browser: Browser | None = None
    _context: BrowserContext | None = None
    _page: Page | None = None

    @classmethod
    async def get_page(cls) -> Page:
        if cls._page is None:
            await cls._start()
        return cls._page

    @classmethod
    async def _start(cls) -> None:
        cls._playwright = await async_playwright().start()
        cls._browser = await cls._playwright.chromium.launch(
            headless=False,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
            ],
        )
        cls._context = await cls._browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        cls._page = await cls._context.new_page()

    @classmethod
    async def close(cls) -> None:
        if cls._page:
            await cls._page.close()
            cls._page = None
        if cls._context:
            await cls._context.close()
            cls._context = None
        if cls._browser:
            await cls._browser.close()
            cls._browser = None
        if cls._playwright:
            await cls._playwright.stop()
            cls._playwright = None
