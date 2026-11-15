/* In-browser demo for the FloodCam project site. Vanilla JS, no dependencies.
 *
 * Data (built by scripts/build_site_assets.py):
 *   frames.json      68 real USGS frames with the held-out reader estimate, gauge, baseline and gate outcome
 *   frame_XX.jpg     the frame, plate outlined
 *   strip_XX.png     the rectified 360 x 48 plate strip
 *   resolution.json  the resolution sweep (gated, learned reader vs colour-threshold baseline)
 * Nothing here computes a depth from pixels: values are replayed from stored results.
 */
(function () {
  'use strict';
  var root = document.getElementById('demo-root');
  if (!root) return;
  var base = root.dataset.base;
  var SIGMA_V = parseFloat(root.dataset.sigmaV);
  var SIGMA_G = parseFloat(root.dataset.sigmaG);
  var C = { gauge: '#0b7285', reader: '#d9480f', base: '#9aa0a8', grid: '#e6e1e8', text: '#5c5f66',
            model: '#2c5282', rej: '#b8bcc4' };
  var DPR = window.devicePixelRatio || 1;

  function $(id) { return document.getElementById(id); }
  function fmt(x, d) { return (x === null || x === undefined || isNaN(x)) ? '–' : x.toFixed(d === undefined ? 1 : d); }
  function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

  function setupCanvas(c, cssW, cssH) {
    c.width = Math.round(cssW * DPR); c.height = Math.round(cssH * DPR);
    c.style.width = cssW + 'px'; c.style.height = 'auto';
    var g = c.getContext('2d'); g.setTransform(DPR, 0, 0, DPR, 0, 0);
    return g;
  }
  function availWidth(c) {
    var el = c.parentElement, cs = window.getComputedStyle(el);
    var pad = (parseFloat(cs.paddingLeft) || 0) + (parseFloat(cs.paddingRight) || 0);
    return Math.max(260, Math.floor(el.clientWidth - pad));
  }

  Promise.all([
    fetch(base + 'frames.json').then(function (r) { return r.json(); }),
    fetch(base + 'resolution.json').then(function (r) { return r.json(); })
  ]).then(function (res) {
    initExplorer(res[0]);
    initResolution(res[1]);
    initStateMachine();
  }).catch(function (err) {
    root.insertAdjacentHTML('afterbegin', '<p class="noscript">Could not load the demo data (' + err + ').</p>');
  });

  /* ------------------------------------------------------------------ shared strip drawing */
  var stripCache = {};
  function loadStrip(i, cb) {
    if (stripCache[i]) { cb(stripCache[i]); return; }
    var im = new Image();
    im.onload = function () { stripCache[i] = im; cb(im); };
    im.onerror = function () { cb(null); };
    im.src = base + 'strip_' + (i < 10 ? '0' : '') + i + '.png';
  }

  /* ------------------------------------------------------------------------- 1. explorer */
  var state = { i: 0, frames: [], meta: null, stripImg: null, showBase: false, timer: null };

  function initExplorer(data) {
    state.frames = data.frames; state.meta = data.meta;
    var slider = $('ex-slider');
    slider.max = state.frames.length - 1;
    slider.addEventListener('input', function () { goTo(parseInt(slider.value, 10)); });
    $('ex-play').addEventListener('click', togglePlay);
    $('ex-base-toggle').addEventListener('change', function (e) { state.showBase = e.target.checked; drawTimeline(); });
    $('ex-timeline').addEventListener('click', function (e) {
      var r = e.target.getBoundingClientRect();
      goTo(nearestByX(e.clientX - r.left));
    });
    window.addEventListener('resize', function () { drawTimeline(); drawResChart(); drawStateMachine(); });
    goTo(Math.floor(state.frames.length / 2));
  }

  function togglePlay() {
    var btn = $('ex-play');
    if (state.timer) { clearInterval(state.timer); state.timer = null; btn.textContent = 'Play'; return; }
    btn.textContent = 'Pause';
    state.timer = setInterval(function () { goTo((state.i + 1) % state.frames.length); }, 450);
  }

  function goTo(i) {
    state.i = clamp(i, 0, state.frames.length - 1);
    var f = state.frames[state.i];
    $('ex-slider').value = state.i;
    $('ex-img').src = base + 'frame_' + (state.i < 10 ? '0' : '') + state.i + '.jpg';
    $('ex-time').textContent = f.time.replace('T', ' ') + ' UTC · frame ' + (state.i + 1) + ' of ' + state.frames.length;
    $('ex-gauge').textContent = fmt(f.gauge_cm) + ' cm';
    $('ex-base').textContent = fmt(f.baseline_cm) + ' cm';
    if (f.accepted) {
      $('ex-reader').textContent = fmt(f.reader_cm) + ' cm';
      $('ex-err').textContent = fmt(Math.abs(f.reader_cm - f.gauge_cm)) + ' cm';
    } else {
      $('ex-reader').textContent = 'abstains';
      $('ex-err').textContent = '–';
    }
    var badges = [];
    badges.push(f.accepted ? '<span class="badge ok">passed quality gate</span>'
      : '<span class="badge rej">rejected: ' + f.reject_reason + '</span>');
    if (f.is_ir) badges.push('<span class="badge info">infrared night image</span>');
    if (!f.registered) badges.push('<span class="badge info">plate position carried over from an earlier frame</span>');
    if (!f.accepted) badges.push('<span class="badge info">it would have read ' + fmt(f.reader_cm) + ' cm</span>');
    $('ex-status').innerHTML = badges.join(' ');
    loadStrip(state.i, function (im) {
      if (state.i !== parseInt($('ex-slider').value, 10)) return;
      state.stripImg = im;
      drawExplorerStrip(); drawResStrip();
    });
    drawTimeline();
  }

  function cmToY(cm, H) {
    var top = state.meta.plate_top_cm, bot = state.meta.plate_bottom_cm;
    return H * (1 - (clamp(cm, bot, top) - bot) / (top - bot));
  }

  function drawExplorerStrip() {
    var c = $('ex-strip'), f = state.frames[state.i], m = state.meta, S = 1.5, ML = 30;
    var W = Math.round(m.strip_cols * S), H = Math.round(m.strip_rows * S);
    var g = setupCanvas(c, ML + W, H);
    g.clearRect(0, 0, ML + W, H);
    if (state.stripImg) {
      g.imageSmoothingEnabled = true;
      g.drawImage(state.stripImg, ML, 0, W, H);
    }
    g.font = '10px sans-serif'; g.textBaseline = 'middle';
    for (var cm = 10; cm <= m.plate_top_cm; cm += 10) {
      var y = cmToY(cm, H);
      g.fillStyle = C.text; g.textAlign = 'right'; g.fillText(cm, ML - 8, y);
      g.fillRect(ML - 5, y, 5, 1);
    }
    g.fillStyle = C.text; g.textAlign = 'right'; g.fillText('cm', ML - 8, 8);
    function level(cm, color, dash, label) {
      var y = cmToY(cm, H);
      g.save(); g.strokeStyle = color; g.lineWidth = 2; g.setLineDash(dash);
      g.beginPath(); g.moveTo(ML, y); g.lineTo(ML + W, y); g.stroke(); g.restore();
      g.textAlign = 'right'; g.lineWidth = 3; g.strokeStyle = 'rgba(255,255,255,.9)';
      var ty = clamp(y - 8, 8, H - 8);
      g.strokeText(label, ML + W - 3, ty); g.fillStyle = color; g.fillText(label, ML + W - 3, ty);
    }
    level(f.gauge_cm, C.gauge, [], 'gauge ' + fmt(f.gauge_cm));
    if (f.accepted) level(f.reader_cm, C.reader, [5, 3], 'reader ' + fmt(f.reader_cm));
    else level(f.reader_cm, C.rej, [2, 3], 'rejected');
    $('ex-strip-note').textContent = f.has_strip ? '' : 'No plate position available for this frame.';
  }

  function plotGeometry(c, h) {
    var W = availWidth(c), g = setupCanvas(c, W, h);
    return { g: g, W: W, H: h, L: 38, R: 10, T: 8, B: 24 };
  }

  function timeMs(f) { return Date.parse(f.time + 'Z'); }
  var tl = null;
  function drawTimeline() {
    var c = $('ex-timeline'); if (!c || !state.frames.length) return;
    var p = plotGeometry(c, 220), g = p.g, F = state.frames;
    var t0 = timeMs(F[0]), t1 = timeMs(F[F.length - 1]);
    var ymax = 10;
    F.forEach(function (f) { ymax = Math.max(ymax, f.gauge_cm, f.reader_cm, state.showBase ? f.baseline_cm : 0); });
    ymax = Math.ceil(ymax / 10) * 10;
    var pw = p.W - p.L - p.R, ph = p.H - p.T - p.B;
    function X(f) { return p.L + (timeMs(f) - t0) / (t1 - t0) * pw; }
    function Y(v) { return p.T + ph * (1 - v / ymax); }
    tl = { X: X, F: F };
    g.clearRect(0, 0, p.W, p.H);
    g.font = '11px sans-serif'; g.fillStyle = C.text; g.strokeStyle = C.grid; g.lineWidth = 1;
    g.textAlign = 'right'; g.textBaseline = 'middle';
    for (var v = 0; v <= ymax; v += 10) {
      g.beginPath(); g.moveTo(p.L, Y(v)); g.lineTo(p.L + pw, Y(v)); g.stroke(); g.fillText(v, p.L - 6, Y(v));
    }
    g.textAlign = 'center'; g.textBaseline = 'top';
    var d0 = new Date(t0); d0.setUTCHours(0, 0, 0, 0);
    for (var d = d0.getTime(); d <= t1; d += 86400000) {
      if (d < t0) continue;
      var x = p.L + (d - t0) / (t1 - t0) * pw;
      g.beginPath(); g.moveTo(x, p.T); g.lineTo(x, p.T + ph); g.stroke();
      g.fillText(new Date(d).toISOString().slice(5, 10), x, p.T + ph + 6);
    }
    g.save(); g.translate(10, p.T + ph / 2); g.rotate(-Math.PI / 2); g.textAlign = 'center'; g.textBaseline = 'top';
    g.fillText('depth (cm)', 0, -10); g.restore();
    // gauge
    g.strokeStyle = C.gauge; g.lineWidth = 2; g.beginPath();
    F.forEach(function (f, k) { if (k) g.lineTo(X(f), Y(f.gauge_cm)); else g.moveTo(X(f), Y(f.gauge_cm)); });
    g.stroke();
    if (state.showBase) {
      g.fillStyle = C.base;
      F.forEach(function (f) { g.beginPath(); g.arc(X(f), Y(f.baseline_cm), 2.2, 0, 6.3); g.fill(); });
    }
    // reader
    F.forEach(function (f) {
      g.beginPath(); g.arc(X(f), Y(f.reader_cm), 3.2, 0, 6.3);
      if (f.accepted) { g.fillStyle = C.reader; g.fill(); }
      else { g.strokeStyle = C.rej; g.lineWidth = 1.5; g.stroke(); }
    });
    // cursor
    var cur = F[state.i]; g.strokeStyle = '#222'; g.lineWidth = 1; g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(X(cur), p.T); g.lineTo(X(cur), p.T + ph); g.stroke(); g.setLineDash([]);
  }
  function nearestByX(px) {
    var best = 0, bd = 1e9;
    tl.F.forEach(function (f, k) { var d = Math.abs(tl.X(f) - px); if (d < bd) { bd = d; best = k; } });
    return best;
  }

  /* ------------------------------------------------------------------------ 2. resolution */
  var rs = { sweep: [], k: 0 };
  function initResolution(sweep) {
    rs.sweep = sweep;
    var s = $('rs-slider'); s.max = sweep.length - 1;
    s.addEventListener('input', function () { rs.k = parseInt(s.value, 10); updateRes(); });
    updateRes();
  }
  function updateRes() { drawResStrip(); drawResChart(); }

  function downscale(img, w, h) {
    var cur = document.createElement('canvas'); cur.width = img.width; cur.height = img.height;
    cur.getContext('2d').drawImage(img, 0, 0);
    while (cur.width / 2 >= w && cur.height / 2 >= h) {   // halve repeatedly: close to area averaging
      var nx = document.createElement('canvas');
      nx.width = Math.max(w, Math.floor(cur.width / 2)); nx.height = Math.max(h, Math.floor(cur.height / 2));
      var gx = nx.getContext('2d'); gx.imageSmoothingEnabled = true; gx.drawImage(cur, 0, 0, nx.width, nx.height);
      cur = nx;
    }
    var out = document.createElement('canvas'); out.width = w; out.height = h;
    var go = out.getContext('2d'); go.imageSmoothingEnabled = true; go.drawImage(cur, 0, 0, w, h);
    return out;
  }

  function drawResStrip() {
    var c = $('rs-strip'); if (!c || !rs.sweep.length) return;
    var e = rs.sweep[rs.k], f = e.factor, H = 360, gap = 24;
    var nativeW = 48, w = Math.max(4, Math.round(48 / f)), h = Math.max(6, Math.round(360 / f));
    var coarseW = Math.max(8, Math.round(w * H / h));
    var g = setupCanvas(c, nativeW + gap + coarseW, H + 16);
    g.clearRect(0, 0, c.width, c.height);
    g.font = '10px sans-serif'; g.fillStyle = C.text; g.textAlign = 'left'; g.textBaseline = 'top';
    g.fillText('native', 0, H + 3); g.fillText('f = ' + f + ' (' + w + '×' + h + ' px)', nativeW + gap, H + 3);
    if (!state.stripImg) return;
    g.drawImage(state.stripImg, 0, 0, nativeW, H);
    var small = downscale(state.stripImg, w, h);
    g.imageSmoothingEnabled = false;
    g.drawImage(small, nativeW + gap, 0, coarseW, H);
  }

  function modelRmse(s) { return Math.sqrt(s * s * SIGMA_V * SIGMA_V + SIGMA_G * SIGMA_G); }
  function drawResChart() {
    var c = $('rs-chart'); if (!c || !rs.sweep.length) return;
    var e = rs.sweep[rs.k];
    $('rs-label').textContent = 'f = ' + e.factor + ' · ' + fmt(e.cm_per_px, 2) + ' cm per pixel';
    $('rs-s').textContent = fmt(e.cm_per_px, 2) + ' cm/px';
    $('rs-mae').textContent = fmt(e.mae, 2) + ' cm';
    $('rs-rmse').textContent = fmt(e.rmse, 2) + ' (' + fmt(modelRmse(e.cm_per_px), 2) + ')';
    $('rs-base').textContent = fmt(e.baseline_mae, 2) + ' cm';
    var p = plotGeometry(c, 240), g = p.g, pw = p.W - p.L - p.R, ph = p.H - p.T - p.B;
    var xmin = Math.log(0.1), xmax = Math.log(14), ymax = 14;
    function X(s) { return p.L + (Math.log(s) - xmin) / (xmax - xmin) * pw; }
    function Y(v) { return p.T + ph * (1 - v / ymax); }
    g.clearRect(0, 0, p.W, p.H);
    g.font = '11px sans-serif'; g.fillStyle = C.text; g.strokeStyle = C.grid; g.lineWidth = 1;
    g.textAlign = 'right'; g.textBaseline = 'middle';
    for (var v = 0; v <= ymax; v += 2) { g.beginPath(); g.moveTo(p.L, Y(v)); g.lineTo(p.L + pw, Y(v)); g.stroke(); g.fillText(v, p.L - 6, Y(v)); }
    g.textAlign = 'center'; g.textBaseline = 'top';
    [0.16, 0.31, 0.62, 1.2, 2.5, 5, 10].forEach(function (s) { g.fillText(s, X(s), p.T + ph + 6); });
    g.fillText('cm per pixel (log scale)', p.L + pw / 2, p.T + ph + 20);
    function line(key, color, dash, w) {
      g.save(); g.strokeStyle = color; g.lineWidth = w; g.setLineDash(dash); g.beginPath();
      rs.sweep.forEach(function (d, k) { var x = X(d.cm_per_px), y = Y(d[key]); if (k) g.lineTo(x, y); else g.moveTo(x, y); });
      g.stroke(); g.restore();
    }
    line('baseline_mae', C.base, [], 2); line('mae', C.reader, [], 2.5);
    g.save(); g.strokeStyle = C.model; g.setLineDash([5, 4]); g.lineWidth = 1.5; g.beginPath();
    for (var k = 0; k <= 60; k++) {
      var s = Math.exp(xmin + (xmax - xmin) * k / 60), y = Y(modelRmse(s));
      if (k) g.lineTo(X(s), y); else g.moveTo(X(s), y);
    }
    g.stroke(); g.restore();
    rs.sweep.forEach(function (d, kk) {
      g.beginPath(); g.arc(X(d.cm_per_px), Y(d.mae), kk === rs.k ? 6 : 3.2, 0, 6.3);
      g.fillStyle = C.reader; g.fill();
      if (kk === rs.k) { g.strokeStyle = '#222'; g.lineWidth = 1.5; g.stroke(); }
    });
    g.strokeStyle = '#222'; g.lineWidth = 1; g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(X(e.cm_per_px), p.T); g.lineTo(X(e.cm_per_px), p.T + ph); g.stroke(); g.setLineDash([]);
  }

  /* --------------------------------------------------------------- 3. state machine port */
  // Port of floodcam.temporal.filter.TemporalFilter and floodcam.temporal.state_machine.StateMachine.
  var SM = { TRIGGER: 10.5, CLEAR: 9.5, FALLING: -1.0, WINDOW_SAMPLES: 7, TREND_MIN: 10, DT: 5, seed: 7,
             out: null, params: null };
  var PRESETS = {
    typical: { peak: 30, rise: 10, hold: 8, fall: 15, noise: 0.6, persist: 5 },
    spike:   { peak: 25, rise: 1, hold: 1, fall: 1, noise: 0.4, persist: 5 },
    edge:    { peak: 11, rise: 8, hold: 20, fall: 10, noise: 0.8, persist: 5 }
  };
  var STATE_COL = { Normal: '#2f855a', Pending: '#b7791f', Flood: '#c53030', Recovering: '#2c5282' };
  var STATE_BG = { Normal: '#eef7f1', Pending: '#fdf6e3', Flood: '#fdeeee', Recovering: '#eaf0fa' };

  function mulberry32(a) {
    return function () { a |= 0; a = a + 0x6D2B79F5 | 0; var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }
  function gauss(r) { var u = 1 - r(), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function median(a) { var s = a.slice().sort(function (x, y) { return x - y; }), m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }
  function slope(ts, ds) {
    var n = ts.length, mt = 0, md = 0, i;
    for (i = 0; i < n; i++) { mt += ts[i]; md += ds[i]; } mt /= n; md /= n;
    var num = 0, den = 0;
    for (i = 0; i < n; i++) { num += (ts[i] - mt) * (ds[i] - md); den += (ts[i] - mt) * (ts[i] - mt); }
    return den ? num / den : 0;
  }

  function profile(p, t) {
    var m = t / 60, t0 = 3, t1 = t0 + p.rise, t2 = t1 + p.hold, t3 = t2 + p.fall;
    if (m < t0) return 0;
    if (m < t1) return p.peak * (m - t0) / p.rise;
    if (m < t2) return p.peak;
    if (m < t3) return p.peak * (1 - (m - t2) / p.fall);
    return 0;
  }

  function simulate(p) {
    var rnd = mulberry32(SM.seed), dt = SM.DT, win = SM.TREND_MIN * 60;
    var T = (3 + p.rise + p.hold + p.fall + 15) * 60, n = Math.floor(T / dt) + 1;
    var acc = [], st = 'Normal', pendingSince = null, out = [], events = [];
    for (var i = 0; i < n; i++) {
      var t = i * dt, raw = Math.max(0, profile(p, t) + p.noise * gauss(rnd));
      acc.push([t, raw]); acc = acc.filter(function (x) { return x[0] >= t - win; });
      var filtered = median(acc.slice(-SM.WINDOW_SAMPLES).map(function (x) { return x[1]; }));
      var rate = null;
      if (acc.length >= 3 && (acc[acc.length - 1][0] - acc[0][0]) >= 0.5 * win) {
        rate = slope(acc.map(function (x) { return x[0] - acc[0][0]; }), acc.map(function (x) { return x[1]; })) * 600;
      }
      var tr = null;
      if (st === 'Normal') {
        if (filtered >= SM.TRIGGER) { st = 'Pending'; pendingSince = t; tr = ['Normal', 'Pending', 'filtered depth ' + fmt(filtered, 1) + ' cm reached the 10.5 cm trigger']; }
      } else if (st === 'Pending') {
        if (filtered < SM.CLEAR) { st = 'Normal'; pendingSince = null; tr = ['Pending', 'Normal', 'depth fell below the 9.5 cm clear level before the persistence time was up']; }
        else if (pendingSince !== null && (t - pendingSince) >= p.persist * 60 && rate !== null && rate > SM.FALLING) {
          st = 'Flood'; tr = ['Pending', 'Flood', 'stayed above 9.5 cm for ' + p.persist + ' min and the trend is flat or rising (' + fmt(rate, 1) + ' cm/10 min)']; pendingSince = null;
        }
      } else if (st === 'Flood') {
        if (rate !== null && rate <= SM.FALLING) { st = 'Recovering'; tr = ['Flood', 'Recovering', 'level is falling (' + fmt(rate, 1) + ' cm/10 min)']; }
      } else if (st === 'Recovering') {
        if (filtered < SM.CLEAR) { st = 'Normal'; tr = ['Recovering', 'Normal', 'depth fell below the 9.5 cm clear level']; }
        else if (rate !== null && rate > SM.FALLING && filtered >= SM.CLEAR) { st = 'Flood'; tr = ['Recovering', 'Flood', 'level is rising again (' + fmt(rate, 1) + ' cm/10 min)']; }
      }
      if (tr) events.push({ t: t, from: tr[0], to: tr[1], why: tr[2] });
      out.push({ t: t, raw: raw, filtered: filtered, rate: rate, state: st });
    }
    return { out: out, events: events, T: T };
  }

  function readParams() {
    return { peak: parseFloat($('sm-peak').value), rise: parseFloat($('sm-rise').value), hold: parseFloat($('sm-hold').value),
             fall: parseFloat($('sm-fall').value), noise: parseFloat($('sm-noise').value), persist: parseFloat($('sm-persist').value) };
  }
  function setParams(p) {
    Object.keys(p).forEach(function (k) { $('sm-' + k).value = p[k]; });
  }
  function initStateMachine() {
    var legend = $('sm-legend');
    legend.innerHTML = Object.keys(STATE_COL).map(function (s) {
      return '<span><i style="background:' + STATE_COL[s] + '"></i>' + s + '</span>'; }).join('') +
      '<span><i style="background:#c9ccd2"></i>raw depth</span><span><i style="background:#222"></i>filtered (median of 7)</span>';
    ['peak', 'rise', 'hold', 'fall', 'noise', 'persist'].forEach(function (k) {
      $('sm-' + k).addEventListener('input', function () { recompute(); });
    });
    document.querySelectorAll('[data-preset]').forEach(function (b) {
      b.addEventListener('click', function () { setParams(PRESETS[b.dataset.preset]); recompute(); });
    });
    $('sm-reseed').addEventListener('click', function () { SM.seed = (SM.seed * 31 + 11) % 100000; recompute(); });
    $('sm-scrub').addEventListener('input', function () { drawStateMachine(); });
    setParams(PRESETS.typical); recompute();
  }
  function recompute() {
    var p = readParams(); SM.params = p;
    ['peak', 'rise', 'hold', 'fall', 'noise', 'persist'].forEach(function (k) { $('sm-' + k + '-v').textContent = p[k]; });
    SM.sim = simulate(p);
    $('sm-scrub').max = SM.sim.out.length - 1;
    if (parseInt($('sm-scrub').value, 10) > SM.sim.out.length - 1) $('sm-scrub').value = SM.sim.out.length - 1;
    var log = $('sm-log');
    log.innerHTML = SM.sim.events.length ? SM.sim.events.map(function (e) {
      return '<div><b>' + fmt(e.t / 60, 1) + ' min</b> &middot; ' + e.from + ' &rarr; ' + e.to + ': ' + e.why + '</div>'; }).join('')
      : '<div>No state change: the filtered depth never reached the 10.5 cm trigger.</div>';
    drawStateMachine();
  }
  function drawStateMachine() {
    var c = $('sm-chart'); if (!c || !SM.sim) return;
    var p = plotGeometry(c, 240), g = p.g, o = SM.sim.out, T = SM.sim.T, pw = p.W - p.L - p.R, ph = p.H - p.T - p.B;
    var ymax = 10;
    o.forEach(function (d) { ymax = Math.max(ymax, d.raw); });
    ymax = Math.ceil((ymax + 2) / 10) * 10;
    function X(t) { return p.L + t / T * pw; }
    function Y(v) { return p.T + ph * (1 - v / ymax); }
    g.clearRect(0, 0, p.W, p.H);
    var i = 0;
    while (i < o.length) {                        // state bands
      var j = i; while (j + 1 < o.length && o[j + 1].state === o[i].state) j++;
      g.fillStyle = STATE_BG[o[i].state]; g.fillRect(X(o[i].t), p.T, X(o[Math.min(j + 1, o.length - 1)].t) - X(o[i].t), ph);
      i = j + 1;
    }
    g.font = '11px sans-serif'; g.fillStyle = C.text; g.strokeStyle = C.grid; g.lineWidth = 1;
    g.textAlign = 'right'; g.textBaseline = 'middle';
    for (var v = 0; v <= ymax; v += 10) { g.beginPath(); g.moveTo(p.L, Y(v)); g.lineTo(p.L + pw, Y(v)); g.stroke(); g.fillText(v, p.L - 6, Y(v)); }
    g.textAlign = 'center'; g.textBaseline = 'top';
    for (var m = 0; m * 60 <= T; m += 10) g.fillText(m, X(m * 60), p.T + ph + 6);
    g.fillText('minutes', p.L + pw / 2, p.T + ph + 20);
    g.save(); g.setLineDash([4, 3]); g.strokeStyle = C.model; g.lineWidth = 1;
    [SM.TRIGGER, SM.CLEAR].forEach(function (th) { g.beginPath(); g.moveTo(p.L, Y(th)); g.lineTo(p.L + pw, Y(th)); g.stroke(); });
    g.restore();
    g.strokeStyle = '#c9ccd2'; g.lineWidth = 1; g.beginPath();
    o.forEach(function (d, k) { if (k) g.lineTo(X(d.t), Y(d.raw)); else g.moveTo(X(d.t), Y(d.raw)); }); g.stroke();
    g.strokeStyle = '#222'; g.lineWidth = 2; g.beginPath();
    o.forEach(function (d, k) { if (k) g.lineTo(X(d.t), Y(d.filtered)); else g.moveTo(X(d.t), Y(d.filtered)); }); g.stroke();
    var k = parseInt($('sm-scrub').value, 10), cur = o[Math.min(k, o.length - 1)];
    g.strokeStyle = '#222'; g.lineWidth = 1; g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(X(cur.t), p.T); g.lineTo(X(cur.t), p.T + ph); g.stroke(); g.setLineDash([]);
    var pill = $('sm-state'); pill.textContent = cur.state; pill.style.background = STATE_COL[cur.state];
    $('sm-now').textContent = 't = ' + fmt(cur.t / 60, 1) + ' min · filtered ' + fmt(cur.filtered, 1) + ' cm · rate ' +
      (cur.rate === null ? 'not yet defined' : fmt(cur.rate, 1) + ' cm/10 min');
  }
})();
