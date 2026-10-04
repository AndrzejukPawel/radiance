/* dashcheck -- run the dashboard's own script against real /stats payloads and report what a
 * reader would call a bug: NaN, undefined, Infinity, or a throw.
 *
 * WHY THIS AND NOT A BROWSER. The dashboard is one self-contained page driven by one endpoint, so
 * the interesting failures are arithmetic rather than layout -- a rate differenced over a zero
 * interval, a panel whose model has no expert plane, a counter that went backwards. Those need the
 * real payloads, which a browser adds nothing to, and they need a SEQUENCE of them, which is
 * awkward to arrange in one.
 *
 * AND poll() SWALLOWS EVERYTHING update() THROWS, marking the dot stale and nothing else. That is
 * the right behaviour for a server that stopped answering and the wrong one for a page that
 * cannot render what it was sent: the two look identical from outside. So this intercepts the
 * exception where it is caught, and reads the dot afterwards -- "dot stale" means it threw.
 *
 * AND A FIGURE CAN BE WRONG WITHOUT BEING NaN. The page formats every number through one block of
 * functions (three significant figures rounded before the unit is chosen, binary sizes, decimal
 * rates, SI prefixes, shares that never round onto 0% or 100%), so those functions are run against
 * a table of boundary cases first, and every text the polls write is then searched for the shapes
 * that rule exists to prevent: a raw float, a four-digit figure in front of a unit or a prefix the
 * next rung replaces, a locale's thousands separator.
 *
 *   node tools/dashcheck.js <saved dashboard html> <json array of /stats payloads>
 *
 * Take the payloads UNDER LOAD. Idle ones exercise none of the differencing, and every rate on the
 * page is a difference.
 */
const fs = require('fs');
const html = fs.readFileSync(process.argv[2], 'utf8');
const payloads = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));

const i0 = html.lastIndexOf('<script>');
const i1 = html.lastIndexOf('</script>');
if (i0 < 0 || i1 < 0) { console.log('NO SCRIPT'); process.exit(2); }
let js = html.slice(i0 + 8, i1);

const writes = [];
const mkEl = (id) => {
  const el = {
    id, _text: '', hidden: false, className: '', innerHTML: '', value: '',
    style: new Proxy({}, { get: () => '', set: () => true }),
    dataset: {}, children: [], scrollTop: 0, scrollHeight: 0, offsetWidth: 100, offsetHeight: 20,
    clientWidth: 800, clientHeight: 400, width: 0, height: 0,
    getContext: () => ({ createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h }),
                         putImageData(){}, clearRect(){}, fillRect(){}, drawImage(){},
                         beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}, fill(){},
                         save(){}, restore(){}, translate(){}, scale(){}, setTransform(){},
                         fillText(){}, measureText: () => ({ width: 10 }) }),
    classList: { add(){}, remove(){}, toggle(){}, contains(){ return false; } },
    appendChild(c){ this.children.push(c); return c; },
    removeChild(){}, remove(){}, insertBefore(c){ this.children.push(c); return c; },
    setAttribute(){}, getAttribute(){ return null; }, removeAttribute(){},
    addEventListener(){}, querySelector(){ return mkEl('q'); },
    querySelectorAll(){ return []; }, getBoundingClientRect(){ return {width:100,height:20,top:0,left:0}; },
    closest(){ return null; }, focus(){}, blur(){}, click(){},
  };
  el.parentNode = {
    style: new Proxy({}, { get: () => '', set: () => true }),
    classList: { add(){}, remove(){}, toggle(){}, contains(){ return false; } },
    querySelector: () => mkEl(id + '>sub'), querySelectorAll: () => [],
    appendChild(){}, removeChild(){}, parentNode: null, children: [],
    clientWidth: 800, clientHeight: 400,
  };
  el.parentElement = el.parentNode;
  /* A REAL ELEMENT HAS A TEXT NODE IN IT, and the page writes through it: setMetric sets
   * firstChild.nodeValue rather than textContent. A stub without one throws inside the page's
   * own try/catch and reads exactly like the bug it is meant to find. */
  const tn = { nodeType: 3, _v: '' };
  Object.defineProperty(tn, 'nodeValue', {
    get(){ return tn._v; },
    set(v){ tn._v = String(v); writes.push([id + '#text', String(v)]); },
  });
  el.firstChild = tn; el.lastChild = tn;
  /* The page indexes childNodes past what a stub would hold -- the sparklines address one
   * path per series. Draw them on demand rather than guessing how many there are. */
  el.childNodes = new Proxy([tn], {
    get(t, p) {
      if (typeof p === 'string' && /^[0-9]+$/.test(p) && t[p] === undefined) t[p] = mkEl(id + '>child' + p);
      return t[p];
    },
  });
  Object.defineProperty(el, 'textContent', {
    get(){ return el._text; },
    set(v){ el._text = String(v); writes.push([id, String(v)]); },
  });
  return el;
};
const els = new Map();
const get = (id) => { if (!els.has(id)) els.set(id, mkEl(id)); return els.get(id); };
global.document = {
  hidden: false,
  getElementById: get,
  createElement: (t) => mkEl('<' + t + '>'),
  createDocumentFragment: () => mkEl('#frag'),
  createTextNode: (t) => { const n = mkEl('#text'); n.textContent = t; return n; },
  createElementNS: (ns, t) => mkEl('<' + t + '>'),
  querySelector: () => mkEl('q'), querySelectorAll: () => [],
  addEventListener(){}, body: mkEl('body'), documentElement: mkEl('html'),
};
global.window = global;
global.location = { hash: '', href: 'http://x/', pathname: '/' };
global.performance = { now: () => Date.now() };
global.requestAnimationFrame = (f) => f();
global.setInterval = () => 0;
global.setTimeout = (f) => { try { f(); } catch (e) {} return 0; };
global.navigator = { userAgent: 'node', clipboard: { writeText(){} } };
global.getComputedStyle = () => new Proxy({}, { get: () => '' });
global.EventSource = function(){ this.addEventListener = () => {}; this.close = () => {}; };
global.matchMedia = () => ({ matches: false, addEventListener(){}, addListener(){} });

let call = 0;
global.fetch = async (u) => ({
  ok: true, status: 200,
  json: async () => {
    if (String(u).indexOf('stats') >= 0) return payloads[Math.min(call++, payloads.length - 1)];
    return {};
  },
  text: async () => '',
});

let threw = null;
/* poll() swallows whatever update() throws and only marks the dot stale, so the exception
 * has to be intercepted where it is caught or it is gone. */
js = js.replace('$("dot").className = "dot stale";',
                'globalThis.__err = globalThis.__err || e; $("dot").className = "dot stale";');
js += "\n;globalThis.__poll = typeof poll === 'function' ? poll : null;";
js += "\n;globalThis.__fmt = typeof ladder === 'function' ? { hb, hbOf, rate, cnt, cnt3, num3, " +
      "pct, pctn, dur, stepTime } : null;";
try { (0, eval)(js); } catch (e) { threw = e; }

/* THE BOUNDARIES, where a formatter that picks its unit before rounding prints a figure in the
 * unit the next one up exists to replace. Each value sits on or just either side of one. */
const G = 2 ** 30, T = 2 ** 40, NB = '\u202f';
const CASES = [
  ['hb', [0], '0 B'], ['hb', [999], '999 B'], ['hb', [1023], '1.00 KiB'], ['hb', [1536], '1.50 KiB'],
  ['hb', [G - 1], '1.00 GiB'], ['hb', [10.5 * G], '10.5 GiB'], ['hb', [65.2 * T], '65.2 TiB'],
  ['hb', [null], '–'],
  ['hbOf', [2.12 * G, 4.33 * G, ' / '], '2.12 / 4.33 GiB'],
  ['hbOf', [960 * 2 ** 20, 1.2 * G, ' / '], '960 MiB / 1.20 GiB'],
  ['rate', [0], '0 B/s'], ['rate', [999.4], '999 B/s'], ['rate', [999.5], '1.00 kB/s'],
  ['rate', [3.531e9], '3.53 GB/s'], ['rate', [999.96e6], '1.00 GB/s'],
  ['cnt3', [999], '999'], ['cnt3', [1000], '1.00k'], ['cnt3', [9995], '10.0k'],
  ['cnt3', [999950], '1.00M'], ['cnt3', [65694228], '65.7M'],
  ['num3', [0], '0'], ['num3', [0.05309], '0.05'], ['num3', [9.996], '10.0'], ['num3', [99.96], '100'],
  ['num3', [111.177288068], '111'], ['num3', [1234.5], '1.23k'],
  ['cnt', [1234], '1234'], ['cnt', [12345], '12' + NB + '345'], ['cnt', [1967390], '1' + NB + '967' + NB + '390'],
  ['cnt', [null], '–'],
  ['pct', [0], '0.0%'], ['pct', [0.0004], '<0.1%'], ['pct', [0.5], '50.0%'], ['pct', [0.9984], '99.8%'],
  ['pct', [0.99996], '>99.9%'], ['pct', [1], '100.0%'], ['pct', [0.004, 0], '<1%'], ['pct', [0.9984, 0], '>99%'],
  ['pctn', [0.4958, 0], '50'],
  ['dur', [0], '0 ms'], ['dur', [0.5201], '520 ms'], ['dur', [0.9996], '1.00 s'], ['dur', [9.996], '10.0 s'],
  ['dur', [59.96], '1m 00s'], ['dur', [173], '2m 53s'], ['dur', [3599.6], '1h 00m'],
  ['dur', [52274], '14h 31m'], ['dur', [86399], '1d 00h'], ['dur', [200000], '2d 08h'],
];
let fmtBad = 0;
if (!globalThis.__fmt) {
  console.log('  no formatter block in this page; boundary cases skipped');
} else {
  for (const [f, args, want] of CASES) {
    const got = globalThis.__fmt[f](...args);
    if (got !== want) { fmtBad++; console.log('  FORMAT ' + f + '(' + args.join(', ') + ') = "' + got + '", want "' + want + '"'); }
  }
  for (const [ms, n, u] of [[34.56, '34.56', 'ms'], [99.996, '100.0', 'ms'], [999.96, '1.00', 's']]) {
    const got = globalThis.__fmt.stepTime(ms);
    if (got.n !== n || got.u !== u) { fmtBad++; console.log('  FORMAT stepTime(' + ms + ') = ' + JSON.stringify(got)); }
  }
  console.log('  ' + (CASES.length + 3 - fmtBad) + ' of ' + (CASES.length + 3) + ' formatter boundary cases hold');
}

(async () => {
  /* show() hid the dashboard view during startup; poll() returns early when it is hidden,
   * so without this the harness measures nothing and says everything is fine. */
  get('view-dash').hidden = false;
  for (let i = 0; i < payloads.length + 1; i++) {
    try { if (globalThis.__poll) await globalThis.__poll(); } catch (e) { threw = threw || e; }
  }
  /* poll() catches everything update() throws and only marks the dot stale, so a real UI
   * failure is invisible from outside. Read the dot: "dot stale" means it threw. */
  const dot = els.has('dot') ? els.get('dot').className : '(never set)';
  console.log('  dot after last poll: ' + dot);
  if (/stale/.test(dot)) console.log('  UPDATE THREW (dashboard would read as stale, not broken)');
  if (globalThis.__err) console.log('  update() threw: ' +
      (globalThis.__err.stack ? globalThis.__err.stack.split('\n').slice(0,4).join('\n      ') : globalThis.__err));
  if (threw) { console.log('THREW: ' + (threw && threw.stack ? threw.stack.split('\n').slice(0,3).join(' | ') : threw)); }
  const bad = writes.filter(([, v]) => /NaN|undefined|Infinity|\[object Object\]|null/.test(v));
  /* A raw float; four digits in front of a unit or a prefix; a locale's thousands separator. */
  const shape = writes.filter(([, v]) =>
      /\d\.\d{4,}/.test(v) ||
      /\d{4,}(\.\d+)?\s?(k|M|G|T|B|KiB|MiB|GiB|TiB|kB\/s|MB\/s|GB\/s|ms)(?![A-Za-z])/.test(v) ||
      /\d,\d{3}(?!\d)/.test(v));
  if (shape.length) {
    console.log('  BADLY FORMATTED (' + shape.length + '):');
    const seen = new Set();
    for (const [id, v] of shape) { const k = id + '=' + v; if (!seen.has(k)) { seen.add(k); console.log('    #' + id + ' -> "' + v + '"'); } }
  }
  console.log('  ' + writes.length + ' text writes across ' + payloads.length + ' polls');
  if (bad.length) {
    console.log('  BAD VALUES (' + bad.length + '):');
    const seen = new Set();
    for (const [id, v] of bad) { const k = id + '=' + v; if (!seen.has(k)) { seen.add(k); console.log('    #' + id + ' -> "' + v + '"'); } }
  } else {
    console.log('  no NaN / undefined / Infinity written');
  }
  process.exit(threw || bad.length || shape.length || fmtBad ? 1 : 0);
})();
