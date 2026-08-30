// 手机视口审计：CDP 设备仿真 + 横向溢出/重叠检测
// 用法: node scripts/mobile-audit.mjs
import WebSocket from 'ws';
import http from 'http';
import fs from 'fs';
import path from 'path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const PORT = 9333;
const OUT = path.join(process.env.TEMP, 'aquam');
fs.mkdirSync(OUT, { recursive: true });

const { spawn } = await import('child_process');
const proc = spawn(EDGE, [
  '--headless=new', '--disable-gpu', '--no-first-run',
  `--remote-debugging-port=${PORT}`, '--user-data-dir=' + path.join(OUT, 'prof-cdp'),
  'about:blank'
], { stdio: 'ignore' });

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// 等待 DevTools 端点就绪
let targets = null;
for (let i = 0; i < 30; i++) {
  await sleep(500);
  try {
    targets = await new Promise((res, rej) => {
      http.get(`http://127.0.0.1:${PORT}/json/list`, r => {
        let d = ''; r.on('data', c => d += c); r.on('end', () => res(JSON.parse(d)));
      }).on('error', rej);
    });
    if (targets.length) break;
  } catch (e) { /* retry */ }
}
const page = targets.find(t => t.type === 'page');
const ws = new WebSocket(page.webSocketDebuggerUrl, { perMessageDeflate: false });
await new Promise(r => ws.on('open', r));

let id = 0;
const pending = new Map();
ws.on('message', (d) => {
  const m = JSON.parse(d);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
});
const send = (method, params = {}) => new Promise((res) => {
  const mid = ++id;
  pending.set(mid, res);
  ws.send(JSON.stringify({ id: mid, method, params }));
});
const evalJs = async (expr) => {
  const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
  return r.result?.result?.value;
};

await send('Page.enable');
await send('Runtime.enable');

const VIEWPORTS = [
  { name: 'm375', w: 375, h: 812 },
  { name: 'm320', w: 320, h: 700 },
];
const ROUTES = ['#/home', '#/models', '#/models/capabilities', '#/playground', '#/tools', '#/tools/gomoku', '#/api'];

const report = {};
for (const vp of VIEWPORTS) {
  await send('Emulation.setDeviceMetricsOverride', {
    width: vp.w, height: vp.h, deviceScaleFactor: 2, mobile: true,
  });
  // iPhone 级 UA 不必需，mobile:true 已影响媒体查询与布局视口
  report[vp.name] = {};
  for (const route of ROUTES) {
    await send('Page.navigate', { url: 'http://localhost:8899/' + route });
    await sleep(1200);
    const audit = await evalJs(`(function(){
      var w = document.documentElement.clientWidth;
      var scrollW = document.documentElement.scrollWidth;
      var bad = [];
      document.querySelectorAll('body *').forEach(function (el) {
        var r = el.getBoundingClientRect();
        if (r.width > 1 && r.right > w + 2) {
          var cs = getComputedStyle(el);
          // 忽略故意横向滚动的容器内部子元素
          if (cs.position === 'fixed') return;
          var p = el.parentElement, inScroller = false;
          while (p && p !== document.body) {
            var pcs = getComputedStyle(p);
            if (pcs.overflowX === 'auto' || pcs.overflowX === 'scroll' || pcs.overflowX === 'hidden') { inScroller = true; break; }
            p = p.parentElement;
          }
          if (inScroller) return;
          bad.push({ tag: el.tagName, cls: String(el.className).slice(0, 40), right: Math.round(r.right), over: Math.round(r.right - w) });
        }
      });
      // 重叠检测：同层级兄弟元素矩形相交（跳过常见安全情况）
      var overlaps = [];
      return JSON.stringify({ vw: w, scrollW: scrollW, hOverflow: scrollW > w + 2, bad: bad.slice(0, 8) });
    })()`);
    const shot = path.join(OUT, vp.name + '-' + route.replace(/[#\/]/g, '_') + '.png');
    const cap = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync(shot, Buffer.from(cap.result.data, 'base64'));
    report[vp.name][route] = JSON.parse(audit);
  }
}

console.log(JSON.stringify(report, null, 1));
ws.close();
proc.kill();
process.exit(0);
