// Shared site footer — injected into #site-footer placeholder
(function () {
  var isPages = window.location.pathname.indexOf('/pages/') !== -1;
  var p = isPages ? '' : 'pages/';   // pages prefix
  var r = isPages ? '../' : '';      // root prefix (unused for now)

  var html = '<footer class="footer">'
    + '<div class="footer-inner">'
    + '<div class="footer-top">'
    + '<div class="footer-brand">'
    + '<div class="footer-logo">NEST</div>'
    + '<p class="footer-desc">Neural EEG Sequence Transducer &mdash; An open-source framework for decoding brain signals into natural language. Built for researchers and engineers.</p>'
    + '<div class="footer-social-links">'
    + '<a href="https://github.com/wazder/NEST" target="_blank" rel="noopener" class="footer-social-link" aria-label="GitHub">'
    + '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>'
    + '</a>'
    + '</div>'
    + '</div>'
    + '<div class="footer-links">'
    + '<div class="footer-column"><h5>Documentation</h5><ul>'
    + '<li><a href="' + p + 'docs.html">Introduction</a></li>'
    + '<li><a href="' + p + 'docs.html#installation">Installation</a></li>'
    + '<li><a href="' + p + 'docs.html#quickstart">Quick Start</a></li>'
    + '<li><a href="' + p + 'docs.html#api">API Reference</a></li>'
    + '<li><a href="' + p + 'docs.html#training">Training Guide</a></li>'
    + '</ul></div>'
    + '<div class="footer-column"><h5>Resources</h5><ul>'
    + '<li><a href="' + p + 'demo.html">Interactive Demo</a></li>'
    + '<li><a href="' + p + 'download.html">Download Models</a></li>'
    + '<li><a href="' + p + 'research.html">Research Paper</a></li>'
    + '<li><a href="' + p + 'about.html">Architecture</a></li>'
    + '<li><a href="https://github.com/wazder/NEST" target="_blank" rel="noopener">GitHub Repo</a></li>'
    + '</ul></div>'
    + '<div class="footer-column"><h5>Community</h5><ul>'
    + '<li><a href="https://github.com/wazder/NEST/issues" target="_blank" rel="noopener">Bug Reports</a></li>'
    + '<li><a href="https://github.com/wazder/NEST/discussions" target="_blank" rel="noopener">Discussions</a></li>'
    + '<li><a href="' + p + 'contributing.html">Contributing</a></li>'
    + '<li><a href="https://github.com/wazder/NEST/blob/main/LICENSE" target="_blank" rel="noopener">MIT License</a></li>'
    + '</ul></div>'
    + '</div>'
    + '</div>'
    + '<div class="footer-divider"></div>'
    + '<div class="footer-bottom">'
    + '<p class="footer-copyright">&copy; 2026 NEST Project &bull; MIT License</p>'
    + '<div class="footer-bottom-links">'
    + '<a href="' + p + 'research.html">Research</a>'
    + '<a href="' + p + 'contributing.html">Contribute</a>'
    + '<a href="https://github.com/wazder/NEST/blob/main/LICENSE" target="_blank" rel="noopener">License</a>'
    + '</div>'
    + '</div>'
    + '</div>'
    + '</footer>';

  var el = document.getElementById('site-footer');
  if (el) el.outerHTML = html;
})();
