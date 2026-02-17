// NEST Website - Main JavaScript
// nest-eeg.org

// Demo functionality
const demoSentences = [
  "The quick brown fox jumps over the lazy dog.",
  "Neural signals contain rich information about our thoughts.",
  "Brain-computer interfaces are transforming how we interact with technology.",
  "NEST decodes EEG signals with state-of-the-art accuracy.",
  "The future of communication lies in understanding the brain.",
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
  
  const currentSentence = demoSentences[demoIndex];
  charIndex = 0;
  output.textContent = '"';
  
  demoInterval = setInterval(() => {
    if (charIndex < currentSentence.length) {
      output.textContent = '"' + currentSentence.substring(0, charIndex + 1);
      if (charIndex === currentSentence.length - 1) {
        output.textContent += '"';
      }
      charIndex++;
    } else {
      clearInterval(demoInterval);
      isPlaying = false;
      demoIndex = (demoIndex + 1) % demoSentences.length;
    }
  }, 50);
}

function resetDemo() {
  if (demoInterval) {
    clearInterval(demoInterval);
  }
  isPlaying = false;
  charIndex = 0;
  demoIndex = 0;
  
  const output = document.getElementById('demo-output');
  if (output) {
    output.textContent = '"The quick brown fox jumps over the lazy dog..."';
  }
}

// Smooth scroll for anchor links
document.addEventListener('DOMContentLoaded', () => {
  // Handle anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });
  
  // Add scroll effect to header
  const header = document.querySelector('.header');
  if (header) {
    window.addEventListener('scroll', () => {
      if (window.scrollY > 50) {
        header.style.backgroundColor = 'rgba(10, 10, 18, 0.95)';
      } else {
        header.style.backgroundColor = 'rgba(10, 10, 18, 0.9)';
      }
    });
  }
  
  // Animate elements on scroll
  const observerOptions = {
    root: null,
    rootMargin: '0px',
    threshold: 0.1
  };
  
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('animate-fadeInUp');
        observer.unobserve(entry.target);
      }
    });
  }, observerOptions);
  
  // Observe elements
  document.querySelectorAll('.step-card, .feature-card').forEach(el => {
    el.style.opacity = '0';
    observer.observe(el);
  });
  
  // Set active nav link based on current page
  const currentPage = window.location.pathname;
  document.querySelectorAll('.nav-link').forEach(link => {
    const href = link.getAttribute('href');
    if (currentPage.endsWith(href) || (currentPage === '/' && href === '/')) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });
});

// Copy code blocks functionality
function copyCode(button) {
  const codeBlock = button.parentElement.querySelector('pre');
  if (codeBlock) {
    const text = codeBlock.textContent;
    navigator.clipboard.writeText(text).then(() => {
      const originalText = button.textContent;
      button.textContent = 'Copied!';
      setTimeout(() => {
        button.textContent = originalText;
      }, 2000);
    });
  }
}

// Mobile menu toggle
function toggleMobileMenu() {
  const nav = document.querySelector('.nav');
  if (nav) {
    nav.classList.toggle('nav-open');
  }
}

// Docs sidebar navigation
function initDocsSidebar() {
  const sidebarLinks = document.querySelectorAll('.docs-sidebar-links a');
  
  sidebarLinks.forEach(link => {
    link.addEventListener('click', function(e) {
      // Remove active class from all links
      sidebarLinks.forEach(l => l.classList.remove('active'));
      // Add active class to clicked link
      this.classList.add('active');
    });
  });
}

// Initialize docs sidebar if on docs page
if (document.querySelector('.docs-sidebar')) {
  initDocsSidebar();
}

// EEG visualization placeholder animation
function initEEGVisualization() {
  const canvas = document.getElementById('eeg-canvas');
  if (!canvas) return;
  
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  
  let time = 0;
  
  function drawWave() {
    ctx.fillStyle = '#1A1A2E';
    ctx.fillRect(0, 0, width, height);
    
    const channels = 8;
    const channelHeight = height / channels;
    
    for (let c = 0; c < channels; c++) {
      ctx.beginPath();
      ctx.strokeStyle = `hsl(${250 + c * 15}, 70%, 60%)`;
      ctx.lineWidth = 1.5;
      
      for (let x = 0; x < width; x++) {
        const y = channelHeight * (c + 0.5) + 
          Math.sin((x * 0.02) + time + c) * 15 +
          Math.sin((x * 0.05) + time * 2 + c * 2) * 8 +
          (Math.random() - 0.5) * 5;
        
        if (x === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }
    
    time += 0.05;
    requestAnimationFrame(drawWave);
  }
  
  drawWave();
}

// Initialize EEG visualization if canvas exists
document.addEventListener('DOMContentLoaded', initEEGVisualization);
