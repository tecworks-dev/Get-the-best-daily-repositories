const socket = io();

document.addEventListener('DOMContentLoaded', function () {

    const sidebarLinks = document.querySelectorAll('.sidebar-nav-item');
    const mainContent = document.querySelector('main');

    const currentPage = window.location.pathname.split('/')[1];
    sidebarLinks.forEach(link => {
        const page = link.getAttribute('data-page');
        if (page === currentPage || (currentPage === '' && page === 'music')) {
            link.classList.add('active');
            if (currentPage == 'music' || (currentPage === '' && page === 'music')) {
                initializeMusicPage();
            }
        } else {
            link.classList.remove('active');
        }
    });

    sidebarLinks.forEach(link => {
        link.addEventListener('click', function (event) {
            event.preventDefault();

            const page = event.target.getAttribute('data-page');

            sidebarLinks.forEach(link => {
                link.classList.remove('active');
            });
            link.classList.add('active');
            window.history.pushState({}, '', `/${page}`);

            fetch(`/${page}`)
                .then(response => response.text())
                .then(html => {
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(html, 'text/html');
                    const content = doc.querySelector('main').innerHTML;
                    mainContent.innerHTML = '';
                    mainContent.innerHTML = content;

                    const newScripts = doc.querySelectorAll('script');
                    const loadScriptPromises = [];

                    newScripts.forEach(oldScript => {
                        const scriptExists = Array.from(document.querySelectorAll('script')).some(script => script.src === oldScript.src);
                        if (!scriptExists && oldScript.src) {
                            const newScript = document.createElement('script');
                            newScript.src = oldScript.src;
                            newScript.defer = true;
                            const scriptLoadPromise = new Promise((resolve, reject) => {
                                newScript.onload = resolve;
                                newScript.onerror = reject;
                            });
                            document.body.appendChild(newScript);
                            loadScriptPromises.push(scriptLoadPromise);
                        }
                    });
                    Promise.all(loadScriptPromises)
                        .then(() => {
                            if (page === 'music' && typeof initializeMusicPage === 'function') {
                                initializeMusicPage();
                            } else if (page === 'subscriptions' && typeof initializeSubscriptionsPage === 'function') {
                                initializeSubscriptionsPage();
                            } else if (page === 'tasks' && typeof initializeTasksPage === 'function') {
                                initializeTasksPage();
                            }
                        })
                        .catch(error => {
                            console.error('Error loading scripts:', error);
                            mainContent.innerHTML = '<p>Sorry, something went wrong. Please try again later.</p>';
                        });
                })
                .catch(error => {
                    console.error('Error fetching page content:', error);
                    mainContent.innerHTML = '<p>Sorry, something went wrong. Please try again later.</p>';
                });
        });
    });

});

function showToast(header, message) {
    var toastContainer = document.querySelector('.toast-container');
    var toastTemplate = document.getElementById('toast-template').cloneNode(true);
    toastTemplate.classList.remove('d-none');

    toastTemplate.querySelector('.toast-header strong').textContent = header;
    toastTemplate.querySelector('.toast-body').textContent = message;
    toastTemplate.querySelector('.text-muted').textContent = new Date().toLocaleString();

    toastContainer.appendChild(toastTemplate);

    var toast = new bootstrap.Toast(toastTemplate);
    toast.show();

    toastTemplate.addEventListener('hidden.bs.toast', function () {
        toastTemplate.remove();
    });
}

socket.on("new_toast_msg", function (data) {
    showToast(data.title, data.message);
});
