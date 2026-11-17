const { chromium } = require("playwright");

(async () => {
  const outputDir =
    "/Users/yangkaiyan/Documents/Codex/2026-11-17/1-10-sse-com-cn-szse/outputs";
  const browser = await chromium.launch({
    headless: true,
    executablePath:
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1050 },
    deviceScaleFactor: 1,
  });

  await page.goto("http://127.0.0.1:8501", {
    waitUntil: "networkidle",
    timeout: 120000,
  });
  await page.getByText("家电上市公司年报问答", { exact: true }).waitFor({
    timeout: 120000,
  });
  await page.screenshot({
    path: `${outputDir}/page-home.png`,
    fullPage: true,
  });

  await page
    .getByRole("button", {
      name: "12家公司中，2025年营业收入和归母净利润最高的分别是谁？",
    })
    .click();
  await page.waitForTimeout(1200);
  await page.getByRole("button", { name: "检索并回答" }).click();
  await page.getByRole("heading", { name: "结论" }).waitFor({
    timeout: 120000,
  });
  await page.waitForTimeout(1000);
  await page.screenshot({
    path: `${outputDir}/page-answer.png`,
    fullPage: true,
  });

  console.log(await page.title());
  await browser.close();
})();
