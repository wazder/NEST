// NEST — nest-eeg.org shared behaviour
(function () {
  'use strict';

  // ---------- theme ----------
  var root = document.documentElement;
  function applyTheme(t) { root.classList.toggle('dark', t === 'dark'); }
  function storedTheme() { try { return localStorage.getItem('nest-theme'); } catch (e) { return null; } }
  var forced = (location.search.match(/[?&]theme=(light|dark)/) || [])[1];
  var initial = forced || storedTheme() || (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  applyTheme(initial);
  document.addEventListener('click', function (e) {
    var btn = e.target.closest('.theme-toggle');
    if (!btn) return;
    var next = root.classList.contains('dark') ? 'light' : 'dark';
    applyTheme(next);
    try { localStorage.setItem('nest-theme', next); } catch (err) {}
  });

  // ---------- mobile nav ----------
  document.addEventListener('click', function (e) {
    var btn = e.target.closest('.menu-btn');
    var nav = document.getElementById('mobile-nav');
    if (btn && nav) {
      var open = nav.classList.toggle('open');
      btn.setAttribute('aria-expanded', String(open));
      return;
    }
    if (nav && e.target.closest('#mobile-nav a')) nav.classList.remove('open');
  });

  // ---------- active nav ----------
  var path = location.pathname.replace(/\/index\.html$/, '/');
  document.querySelectorAll('.nav a, .mobile-nav a').forEach(function (a) {
    var href = a.getAttribute('href') || '';
    var file = href.split('/').pop();
    var here = path.split('/').pop();
    var isHome = (here === '' || here === 'index.html') && (file === '' || file === 'index.html');
    if (isHome || (file && file !== 'index.html' && here === file)) a.classList.add('active');
  });

  // ---------- reveal ----------
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var revealEls = document.querySelectorAll('.reveal');
  if (reduce || !('IntersectionObserver' in window)) {
    revealEls.forEach(function (el) { el.classList.add('in'); });
  } else {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); } });
    }, { rootMargin: '0px 0px -40px 0px', threshold: 0.05 });
    revealEls.forEach(function (el) { io.observe(el); });
  }

  // ---------- EEG canvas (monochrome, illustrative) ----------
  function eeg(canvas, channels) {
    var ctx = canvas.getContext('2d'), w, h, t = 0, raf;
    function resize() {
      var r = canvas.getBoundingClientRect();
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.round(r.width * dpr));
      canvas.height = Math.max(1, Math.round(r.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      w = r.width; h = r.height;
    }
    function draw() {
      ctx.fillStyle = '#161616'; ctx.fillRect(0, 0, w, h);
      var ch = h / channels;
      for (var c = 0; c < channels; c++) {
        var cy = ch * (c + 0.5), amp = ch * 0.28 + (c % 3) * 2;
        ctx.beginPath();
        ctx.strokeStyle = c % 4 === 0 ? '#a99bf5' : 'rgba(255,255,255,0.55)';
        ctx.lineWidth = 1;
        for (var x = 0; x <= w; x += 2) {
          var s = x * 0.02 + t + c * 0.9;
          var y = cy + Math.sin(s) * amp * 0.6 + Math.sin(s * 2.7 + c) * amp * 0.3 + Math.sin(s * 0.4) * amp * 0.2 + (Math.random() - 0.5) * 1.5;
          x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
      t += reduce ? 0 : 0.035;
      raf = requestAnimationFrame(draw);
    }
    if ('ResizeObserver' in window) new ResizeObserver(resize).observe(canvas); else window.addEventListener('resize', resize);
    resize(); draw();
    document.addEventListener('visibilitychange', function () { if (document.hidden) cancelAnimationFrame(raf); else draw(); });
  }
  document.querySelectorAll('canvas.eeg-canvas').forEach(function (c) { eeg(c, parseInt(c.dataset.channels || '6', 10)); });

  // ---------- typing demo ----------
  var demo = { timer: null, i: 0 };
  var sentences = [
    'presents a good case while failing to provide a reason for us to care',
    'the film is a fascinating and moving character study',
    'he was born in london and studied law at cambridge',
    'the company reported higher earnings in the first quarter'
  ];
  window.nestDemoStart = function () {
    var out = document.getElementById('demo-output'); if (!out || demo.timer) return;
    var s = sentences[demo.i % sentences.length]; var k = 0; out.textContent = '';
    demo.timer = setInterval(function () {
      out.textContent = s.slice(0, ++k);
      if (k >= s.length) { clearInterval(demo.timer); demo.timer = null; demo.i++; }
    }, reduce ? 1 : 38);
  };
  window.nestDemoReset = function () {
    if (demo.timer) clearInterval(demo.timer); demo.timer = null; demo.i = 0;
    var out = document.getElementById('demo-output'); if (out) out.textContent = out.dataset.idle || '';
  };
  document.addEventListener('click', function (e) {
    if (e.target.closest('[data-demo="start"]')) window.nestDemoStart();
    if (e.target.closest('[data-demo="reset"]')) window.nestDemoReset();
  });

  // ---------- copy ----------
  document.addEventListener('click', function (e) {
    var b = e.target.closest('.copy'); if (!b) return;
    var pre = b.closest('.code') && b.closest('.code').querySelector('pre'); if (!pre) return;
    navigator.clipboard.writeText(pre.textContent.trim()).then(function () {
      var o = b.textContent; b.textContent = 'Copied'; setTimeout(function () { b.textContent = o; }, 1500);
    });
  });

  // ---------- docs sidebar ----------
  var side = document.querySelectorAll('.docs-side a');
  var secs = document.querySelectorAll('.doc-section[id]');
  if (side.length && secs.length && 'IntersectionObserver' in window) {
    var so = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        side.forEach(function (a) { a.classList.toggle('active', a.getAttribute('href') === '#' + en.target.id); });
      });
    }, { rootMargin: '-15% 0px -70% 0px' });
    secs.forEach(function (s) { so.observe(s); });
  }

  // ---------- QA harness (only with ?qa=1; used by scripts/qa.py) ----------
  if (/[?&]qa=1/.test(location.search)) {
    window.addEventListener('load', function () {
      setTimeout(function () {
        var vw = document.documentElement.clientWidth;
        var overflow = [];
        var scrollOk = function (el) {
          while (el && el !== document.body) { var o = getComputedStyle(el).overflowX; if (o === 'auto' || o === 'scroll' || o === 'hidden') return true; el = el.parentElement; }
          return false;
        };
        document.querySelectorAll('body *').forEach(function (el) {
          if (el.id === '__qa' || el.closest('#__qa')) return;
          var r = el.getBoundingClientRect(); if (!r.width || !r.height) return;
          if ((r.right > vw + 1 || r.left < -1) && !scrollOk(el)) {
            overflow.push({ tag: el.tagName.toLowerCase(), cls: (el.className && el.className.baseVal === undefined ? String(el.className) : '').slice(0, 40), left: Math.round(r.left), right: Math.round(r.right), text: (el.textContent || '').trim().slice(0, 40) });
          }
        });
        var clipped = [];
        document.querySelectorAll('h1,h2,h3,p,li,td,th,.btn,.label,.stat .v').forEach(function (el) {
          if (el.scrollWidth > el.clientWidth + 2 && getComputedStyle(el).overflowX !== 'visible') clipped.push(el.tagName + ':' + (el.textContent || '').trim().slice(0, 40));
        });
        var imgs = []; document.querySelectorAll('img').forEach(function (i) { if (!i.complete || !i.naturalWidth) imgs.push(i.getAttribute('src')); });
        var res = {
          page: location.pathname.split('/').slice(-2).join('/'), vw: vw,
          docScrollWidth: document.documentElement.scrollWidth,
          horizontalScroll: document.documentElement.scrollWidth > vw + 1,
          overflowCount: overflow.length, overflow: overflow.slice(0, 12), clipped: clipped.slice(0, 12), brokenImages: imgs,
          h1Count: document.querySelectorAll('h1').length,
          fontInter: document.fonts ? document.fonts.check('900 16px Inter') : null,
          title: document.title, dark: root.classList.contains('dark')
        };
        var pre = document.createElement('pre'); pre.id = '__qa'; pre.textContent = JSON.stringify(res); document.body.appendChild(pre);
      }, 1200);
    });
  }
})();
