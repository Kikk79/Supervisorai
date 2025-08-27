from playwright.sync_api import sync_playwright, expect

def verify_log_panel():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto("http://localhost:5000")

        # Wait for the log panel to be visible
        log_panel = page.locator("#log-container")
        expect(log_panel).to_be_visible()

        # Wait for at least one log entry to appear
        # This will wait for the first child div to be present in the log container
        first_log_entry = log_panel.locator("div:first-child")
        expect(first_log_entry).to_be_visible(timeout=10000) # Wait up to 10 seconds

        # Take a screenshot
        page.screenshot(path="jules-scratch/verification/log_panel_verification.png")

        browser.close()

if __name__ == "__main__":
    verify_log_panel()
