document.addEventListener('keydown', function(event) {
  if (event.keyCode === 32) {
    document.body.classList.add('animation-ready');
    document.body.classList.toggle('dark');
  }
});
