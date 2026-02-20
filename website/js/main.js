// NEST Website - Main JavaScript
// nest-eeg.org

// ===== HEADER SCROLL EFFECT =====
(function initHeader() {
  const header = document.querySelector('.header');
  if (!header) return;

  const onScroll = () => {
    header.classList.toggle('scrolled', window.scrollY > 40);
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();

// ===== MOBILE MENU =====
(function initMobileMenu() {
  const btn = document.getElementById('mobile-menu-btn');
  const nav = document.getElementById('mobile-nav');
  if (!btn || !nav) return;

  btn.addEventListener('click', () => {
    const isOpen = nav.classList.toggle('open');
    btn.classList.toggle('open', isOpen);
    btn.setAttribute('aria-expanded', isOpen);
  });

  // Close on nav link click
  nav.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => {
      nav.classList.remove('open');
      btn.classList.remove('open');
      btn.setAttribute('aria-expanded', false);
    });
  });
})();

// ===== ACTIVE NAV LINK =====
(function setActiveNav() {
  const path = window.location.pathname;
  document.querySelectorAll('.nav-link, .mobile-nav .nav-link').forEach(link => {
    const href = link.getAttribute('href') || '';
    const isHome = (path === '/' || path.endsWith('index.html')) && (href === '/' || href === '../index.html' || href === 'index.html');
    const isMatch = !isHome && href !== '/' && href !== '../index.html' && path.includes(href.replace('../', ''));
    link.classList.toggle('active', isHome || isMatch);
  });
})();

// ===== SMOOTH SCROLL FOR ANCHOR LINKS =====
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
});

// ===== SCROLL REVEAL ANIMATIONS =====
(function initReveal() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, {
    root: null,
    rootMargin: '0px 0px -60px 0px',
    threshold: 0.08
  });

  document.querySelectorAll('.reveal').forEach(el => observer.observe(el));
})();

// ===== STAT COUNTER ANIMATION =====
function animateCounter(el) {
  const target = parseFloat(el.dataset.target);
  const isDecimal = el.dataset.decimal === 'true';
  const suffix = el.dataset.suffix || '';
  const duration = 1800;
  const start = performance.now();

  const update = (now) => {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    // Ease out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const value = target * eased;
    el.textContent = (isDecimal ? value.toFixed(2) : Math.round(value)) + suffix;
    if (progress < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

(function initCounters() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        animateCounter(entry.target);
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.5 });

  document.querySelectorAll('[data-target]').forEach(el => observer.observe(el));
})();

// ===== HERO NEURAL NETWORK CANVAS =====
(function initHeroCanvas() {
  const canvas = document.getElementById('hero-canvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  let width, height, nodes, animFrame;

  const NODE_COUNT = 55;
  const CONNECTION_DIST = 160;
  const NODE_SPEED = 0.3;

  function resize() {
    width = canvas.width = canvas.offsetWidth;
    height = canvas.height = canvas.offsetHeight;
  }

  function createNodes() {
    nodes = Array.from({ length: NODE_COUNT }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * NODE_SPEED,
      vy: (Math.random() - 0.5) * NODE_SPEED,
      r: Math.random() * 2 + 1,
      pulse: Math.random() * Math.PI * 2
    }));
  }

  function draw() {
    ctx.clearRect(0, 0, width, height);

    // Update positions
    nodes.forEach(n => {
      n.x += n.vx;
      n.y += n.vy;
      n.pulse += 0.02;

      if (n.x < 0 || n.x > width) n.vx *= -1;
      if (n.y < 0 || n.y > height) n.vy *= -1;
    });

    // Draw connections
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < CONNECTION_DIST) {
          const alpha = (1 - dist / CONNECTION_DIST) * 0.18;
          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.strokeStyle = `rgba(139, 92, 246, ${alpha})`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    }

    // Draw nodes
    nodes.forEach(n => {
      const glow = (Math.sin(n.pulse) + 1) / 2;
      const r = n.r + glow * 1;
      const alpha = 0.3 + glow * 0.4;

      ctx.beginPath();
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(139, 92, 246, ${alpha})`;
      ctx.fill();
    });

    animFrame = requestAnimationFrame(draw);
  }

  const ro = new ResizeObserver(() => {
    resize();
  });
  ro.observe(canvas.parentElement);

  resize();
  createNodes();
  draw();

  // Pause when not visible
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      cancelAnimationFrame(animFrame);
    } else {
      draw();
    }
  });
})();

// ===== EEG MINI CANVAS (landing page demo section) =====
(function initEEGMini() {
  const canvas = document.getElementById('eeg-mini-canvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  let width, height, time = 0, animFrame;

  function resize() {
    width = canvas.width = canvas.offsetWidth;
    height = canvas.height = canvas.offsetHeight;
  }

  const CHANNELS = 6;
  const COLORS = ['#8B5CF6', '#6366F1', '#A855F7', '#10B981', '#22C55E', '#6366F1'];

  function draw() {
    ctx.fillStyle = '#0D0D1A';
    ctx.fillRect(0, 0, width, height);

    const channelH = height / CHANNELS;

    for (let c = 0; c < CHANNELS; c++) {
      const centerY = channelH * (c + 0.5);
      const amp = 18 + (c % 3) * 6;

      ctx.beginPath();
      ctx.strokeStyle = COLORS[c];
      ctx.lineWidth = 1.5;
      ctx.shadowColor = COLORS[c];
      ctx.shadowBlur = 4;

      for (let x = 0; x <= width; x += 2) {
        const t = (x * 0.015) + time + c * 0.8;
        const y = centerY
          + Math.sin(t) * amp
          + Math.sin(t * 2.3 + c) * (amp * 0.4)
          + Math.sin(t * 0.5 + c * 2) * (amp * 0.2)
          + (Math.random() - 0.5) * 3;

        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    time += 0.04;
    animFrame = requestAnimationFrame(draw);
  }

  const ro = new ResizeObserver(resize);
  ro.observe(canvas);

  resize();
  draw();
})();

// ===== EEG FULL CANVAS (demo page) =====
function initEEGVisualization() {
  const canvas = document.getElementById('eeg-canvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  let width, height, time = 0;

  function resize() {
    width = canvas.width = canvas.offsetWidth;
    height = canvas.height = canvas.offsetHeight;
  }

  const CHANNELS = 10;
  const COLORS = [
    '#8B5CF6', '#6366F1', '#A855F7', '#10B981',
    '#22C55E', '#F59E0B', '#EC4899', '#06B6D4',
    '#8B5CF6', '#6366F1'
  ];

  function draw() {
    ctx.fillStyle = '#0A0A14';
    ctx.fillRect(0, 0, width, height);

    const channelH = height / CHANNELS;

    for (let c = 0; c < CHANNELS; c++) {
      const centerY = channelH * (c + 0.5);
      const amp = 12 + (c % 4) * 5;

      ctx.beginPath();
      ctx.strokeStyle = COLORS[c];
      ctx.lineWidth = 1.5;
      ctx.shadowColor = COLORS[c];
      ctx.shadowBlur = 3;

      for (let x = 0; x <= width; x += 2) {
        const t = (x * 0.018) + time + c * 0.9;
        const y = centerY
          + Math.sin(t) * amp
          + Math.sin(t * 2.1 + c) * (amp * 0.35)
          + Math.sin(t * 0.7) * (amp * 0.15)
          + (Math.random() - 0.5) * 2;

        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    time += 0.035;
    requestAnimationFrame(draw);
  }

  window.addEventListener('resize', resize);
  resize();
  draw();
}

document.addEventListener('DOMContentLoaded', initEEGVisualization);

// ===== DEMO FUNCTIONALITY =====
const demoSentences = [
  "The neural signals reveal a structured thought pattern.",
  "Brain-computer interfaces are transforming how we communicate.",
  "NEST decodes EEG signals with state-of-the-art accuracy.",
  "The quick brown fox jumps over the lazy dog.",
  "Cognitive load varies significantly between reading tasks.",
];

let demoIndex = 0;
let charIndex = 0;
let isPlaying = false;
let demoInterval = null;

function startDemo() {
  if (isPlaying) return;
  isPlaying = true;

  const output = document.getElementById('demo-output');
  if (!output) return;

  const sentence = demoSentences[demoIndex];
  charIndex = 0;
  output.textContent = '';

  demoInterval = setInterval(() => {
    if (charIndex < sentence.length) {
      output.textContent = sentence.substring(0, charIndex + 1);
      charIndex++;
    } else {
      clearInterval(demoInterval);
      isPlaying = false;
      demoIndex = (demoIndex + 1) % demoSentences.length;
    }
  }, 40);
}

function resetDemo() {
  if (demoInterval) clearInterval(demoInterval);
  isPlaying = false;
  charIndex = 0;
  demoIndex = 0;

  const output = document.getElementById('demo-output');
  if (output) output.textContent = 'Click "Start Demo" to begin decoding...';
}

// ===== CODE COPY =====
function copyCode(btn) {
  const pre = btn.closest('.code-block')?.querySelector('pre');
  if (!pre) return;

  navigator.clipboard.writeText(pre.textContent.trim()).then(() => {
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = orig; }, 2000);
  });
}

// ===== DOCS SIDEBAR =====
(function initDocsSidebar() {
  if (!document.querySelector('.docs-sidebar')) return;

  const links = document.querySelectorAll('.docs-sidebar-links a');
  const sections = document.querySelectorAll('.docs-content h2[id], .docs-content h3[id]');

  if (!sections.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        links.forEach(l => l.classList.remove('active'));
        const active = document.querySelector(`.docs-sidebar-links a[href="#${entry.target.id}"]`);
        if (active) active.classList.add('active');
      }
    });
  }, { rootMargin: '-20% 0px -70% 0px' });

  sections.forEach(s => observer.observe(s));

  links.forEach(link => {
    link.addEventListener('click', function (e) {
      links.forEach(l => l.classList.remove('active'));
      this.classList.add('active');
    });
  });
})();
