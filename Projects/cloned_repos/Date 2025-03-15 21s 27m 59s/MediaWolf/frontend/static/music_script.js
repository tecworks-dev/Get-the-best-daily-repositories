let selectedType = "track"

function initializeMusicPage() {
    const searchTabLink = document.getElementById('pills-music-search-tab');
    const recommendationsTabLink = document.getElementById('pills-music-recommendations-tab');
    const lidarrWantedTabLink = document.getElementById('pills-lidarr-wanted-tab');

    searchTabLink.addEventListener('shown.bs.tab', function () {
        setupMusicSearchTab();
        storeActiveMusicTab('pills-music-search-tab');
    });

    recommendationsTabLink.addEventListener('shown.bs.tab', function () {
        setupMusicRecommendationsTab();
        storeActiveMusicTab('pills-music-recommendations-tab');
    });

    lidarrWantedTabLink.addEventListener('shown.bs.tab', function () {
        setupLidarrWantedTab();
        storeActiveMusicTab('pills-lidarr-wanted-tab');
    });

    const lastActiveMusicTab = localStorage.getItem('lastActiveMusicTab');

    if (lastActiveMusicTab) {
        const lastActiveMusicTabElement = document.getElementById(lastActiveMusicTab);
        if (lastActiveMusicTabElement) {
            new bootstrap.Tab(lastActiveMusicTabElement).show();

            if (lastActiveMusicTab === 'pills-music-search-tab') {
                setupMusicSearchTab();
            } else if (lastActiveMusicTab === 'pills-music-recommendations-tab') {
                setupMusicRecommendationsTab();
            } else if (lastActiveMusicTab === 'pills-lidarr-wanted-tab') {
                setupLidarrWantedTab();
            }
        }
    } else {
        setupMusicSearchTab();
    }


}

function storeActiveMusicTab(tabId) {
    localStorage.setItem('lastActiveMusicTab', tabId);
}

// Search Tab
function setupMusicSearchTab() {
    var searchButton = document.getElementById('music-search-button');
    var searchInput = document.getElementById('music-search-input');

    searchButton.addEventListener('click', initiateSearch);
    searchInput.addEventListener('keydown', function (event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            initiateSearch();
        }
    });
}

function handleDownloadClick(event) {
    const button = event.target;

    button.disabled = true;
    button.classList.remove('btn-primary');
    button.classList.add('btn-secondary');
    button.classList.add('disabled');
    button.innerText = 'Added';

    const trackUrl = button.getAttribute('data-url');
    const card = button.closest('.card');

    if (!trackUrl) {
        alert('No URL found!');
        return;
    }

    const itemData = {
        type: card.querySelector('.type')?.textContent.trim().toLowerCase(),
        name: card.querySelector('.name')?.textContent.trim(),
        artist: card.querySelector('.artist')?.textContent.trim() || null,
        url: trackUrl
    };

    socket.emit('spotify_download_item', itemData);
}

function populateSpotifyTemplate(type, data) {
    let templateId;
    let element;

    if (type === 'track') {
        templateId = 'spotify-track-item-template';
    } else if (type === 'album') {
        templateId = 'spotify-album-item-template';
    } else if (type === 'artist') {
        templateId = 'spotify-artist-item-template';
    } else if (type === 'playlist') {
        templateId = 'spotify-playlist-item-template';
    }

    const template = document.getElementById(templateId);
    const clone = document.importNode(template.content, true);

    const downloadButton = clone.querySelector('.download');

    if (type === 'track') {
        clone.querySelector('.track-img').src = data.image || 'https://picsum.photos/300';
        clone.querySelector('.name').textContent = data.name;
        clone.querySelector('.artist').textContent = data.artist;
        clone.querySelector('.download').href = data.url;
        clone.querySelector('.download').setAttribute('data-url', data.url);
    } else if (type === 'album') {
        clone.querySelector('.album-img').src = data.image || 'https://picsum.photos/300';
        clone.querySelector('.name').textContent = data.name;
        clone.querySelector('.artist').textContent = data.artist;
        clone.querySelector('.download').href = data.url;
        clone.querySelector('.download').setAttribute('data-url', data.url);
    } else if (type === 'artist') {
        clone.querySelector('.artist-img').src = data.image || 'https://picsum.photos/300';
        clone.querySelector('.name').textContent = data.name;
        clone.querySelector('.followers').textContent = `${data.followers} Followers`;
        clone.querySelector('.download').href = data.url;
        clone.querySelector('.download').setAttribute('data-url', data.url);
    } else if (type === 'playlist') {
        clone.querySelector('.playlist-img').src = data.image || 'https://picsum.photos/300';
        clone.querySelector('.name').textContent = data.name;
        clone.querySelector('.owner').textContent = data.owner;
        clone.querySelector('.download').href = data.url;
        clone.querySelector('.download').setAttribute('data-url', data.url);
    }

    element = document.getElementById('results-section');
    element.appendChild(clone);

    downloadButton.addEventListener('click', function (event) {
        event.preventDefault();
        handleDownloadClick(event);
    });
}

function changeUI(reqState) {
    var spinnerBorder = document.getElementById('music-spinner-border');
    var searchDropdown = document.getElementById('music-search-dropdown');
    var searchButton = document.getElementById('music-search-button');
    var searchInput = document.getElementById('music-search-input');

    if (reqState === "busy") {
        searchInput.disabled = true;
        searchButton.disabled = true;
        searchDropdown.disabled = true;
        if (spinnerBorder) {
            spinnerBorder.style.display = 'inline-block';
        }
    } else {
        if (spinnerBorder) {
            spinnerBorder.style.display = 'none';
        }
        searchInput.disabled = false;
        searchButton.disabled = false;
        searchDropdown.disabled = false;
    }
}

function initiateSearch() {
    changeUI("busy");
    var searchInput = document.getElementById('music-search-input');
    const searchText = searchInput.value.trim();
    socket.emit('search_spotify', { query: searchText, type: selectedType });
}

function updateSearchSelection(option) {
    selectedType = option.toLowerCase();
    document.getElementById("music-search-button").innerText = `Search for ${option}`;
}


// Recommendations Tab
function setupMusicRecommendationsTab() {
    socket.emit("load_music_recommendations");
    var getRecommendationsButton = document.getElementById('get-recommendations-button');
    getRecommendationsButton.addEventListener('click', getRecommendations)

    setTimeout(() => {
        ["min-listeners", "min-play-count"].forEach(id => {
            let element = document.getElementById(id);
            if (element) {
                element.addEventListener("input", function () {
                    updateStep(this);
                });
            }
        });
    }, 500);
}

function appendArtists(artists) {
    var artistRow = document.getElementById('artist-row');
    var template = document.getElementById('artist-template');

    artists.forEach(function (artist) {
        var clone = document.importNode(template.content, true);
        var artistCol = clone.querySelector('#artist-column');

        artistCol.querySelector('.card-title').textContent = artist.name;
        artistCol.querySelector('.genre').textContent = artist.genre;
        artistCol.querySelector('.card-img-top').src = artist.image;
        artistCol.querySelector('.card-img-top').alt = artist.name;
        artistCol.querySelector('.add-to-lidarr-btn').addEventListener('click', function () {
            addToLidarr(artist.name);
        });
        artistCol.querySelector('.get-overview-btn').addEventListener('click', function () {
            overviewReq(artist);
        });
        artistCol.querySelector('.dismiss-artist-btn').addEventListener('click', function (event) {
            dismissArtist(event, artist);
        });
        artistCol.querySelector('.listeners').textContent = artist.listeners;
        artistCol.querySelector('.play-count').textContent = artist.play_count;

        var addButton = artistCol.querySelector('.add-to-lidarr-btn');
        if (artist.status === "Added" || artist.status === "Already in Lidarr") {
            artistCol.querySelector('.card-body').classList.add('status-green');
            addButton.classList.remove('btn-primary');
            addButton.classList.add('btn-secondary');
            addButton.disabled = true;
            addButton.textContent = artist.status;
        } else if (artist.status === "Failed to Add" || artist.status === "Invalid Path") {
            artistCol.querySelector('.card-body').classList.add('status-red');
            addButton.classList.remove('btn-primary');
            addButton.classList.add('btn-danger');
            addButton.disabled = true;
            addButton.textContent = artist.status;
        } else {
            artistCol.querySelector('.card-body').classList.add('status-blue');
        }
        artistRow.appendChild(clone);
    });
}

function addToLidarr(artistName) {
    if (socket.connected) {
        socket.emit('add_artist_to_lidarr', encodeURIComponent(artistName));
    }
    else {
        showToast("Connection Lost", "Please reload to continue.");
    }
}

function dismissArtist(event, artist) {
    if (socket.connected) {
        socket.emit('dismiss_artist', encodeURIComponent(artist.name));
        const artistColumn = event.currentTarget.closest('#artist-column');

        if (artistColumn) {
            artistColumn.style.transition = 'opacity 1.5s';
            artistColumn.style.opacity = '0';

            setTimeout(() => artistColumn.remove(), 1500);
        }
    }
    else {
        showToast("Connection Lost", "Please reload to continue.");
    }
}

let isLoading = false;
let currentPage = 1;

function getRecommendations(refresh = false) {
    if (isLoading) return;

    isLoading = true;
    if (refresh) currentPage = 1;

    const data = {
        selected_artist: document.getElementById("artist-select").value || "all",
        sort_by: document.getElementById("sort-select").value || "random",
        min_play_count: parseInt(document.getElementById("min-play-count").value) || null,
        min_listeners: parseInt(document.getElementById("min-listeners").value) || null,
        num_results: 10,
        page: currentPage,
    };

    socket.emit("refresh_music_recommendations", data);


}

function loadMoreRecommendations() {
    getRecommendations(false);
}

function updateStep(input) {
    let value = parseInt(input.value);
    input.step = value >= 10_000_000 ? 1_000_000 :
        value >= 1_000_000 ? 100_000 :
            value >= 100_000 ? 10_000 : 1_000;
}

function overviewReq(artist) {
    const scrollbarWidth = window.innerWidth - document.documentElement.clientWidth;
    document.body.style.overflow = 'hidden';
    document.body.style.paddingRight = `${scrollbarWidth}px`;

    var modalTitle = document.getElementById('bio-modal-title');
    var modalBody = document.getElementById('modal-body');
    modalTitle.textContent = artist.name;
    modalBody.innerHTML = DOMPurify.sanitize(artist.overview);

    var modal = new bootstrap.Modal(document.getElementById('bio-modal-modal'));
    modal.show();

    modal._element.addEventListener('hidden.bs.modal', function () {
        document.body.style.overflow = 'auto';
        document.body.style.paddingRight = '0';
    });
}

// Lidarr Wanted Tab
function setupLidarrWantedTab() {
    var lidarrSpinner = document.getElementById('lidarr-spinner');
    var lidarrTable = document.getElementById('lidarr-table').getElementsByTagName('tbody')[0];
    var getWantedLidarr = document.getElementById('get-lidarr-wanted-btn');
    var selectAllCheckbox = document.getElementById("select-all-checkbox");

    selectAllCheckbox.addEventListener("change", function () {
        var isChecked = this.checked;
        var checkboxes = document.querySelectorAll('input[name="lidarr_item"]');
        checkboxes.forEach(function (checkbox) {
            checkbox.checked = isChecked;
        });
    });

    getWantedLidarr.addEventListener('click', function () {
        getWantedLidarr.disabled = true;
        lidarrSpinner.classList.remove('d-none');
        lidarrTable.innerHTML = '';
        socket.emit("lidarr_get_wanted");
    });


}

document.addEventListener('DOMContentLoaded', function () {
    initializeMusicPage();
});


socket.on('spotify_search_results', function (data) {
    changeUI("ready");
    const resultsSection = document.getElementById('results-section');
    resultsSection.innerHTML = '';

    if (data && data.results) {
        for (let category in data.results) {
            let items = data.results[category];

            if (Array.isArray(items) && items.length > 0) {
                items.forEach(item => {
                    populateSpotifyTemplate(item.type, item);
                });
            } else {
                const noResultsMessage = document.createElement('p');
                noResultsMessage.textContent = 'No results found';
                resultsSection.appendChild(noResultsMessage);
            }
        }
    }
});

socket.on("refresh_artist", (artist) => {
    var artistCards = document.querySelectorAll('#artist-column');
    artistCards.forEach(function (card) {
        var cardBody = card.querySelector('.card-body');
        var cardArtistName = cardBody.querySelector('.card-title').textContent.trim();

        if (cardArtistName === artist.name) {
            cardBody.classList.remove('status-green', 'status-red', 'status-blue');

            var addButton = cardBody.querySelector('.add-to-lidarr-btn');

            if (artist.status === "Added" || artist.status === "Already in Lidarr") {
                cardBody.classList.add('status-green');
                addButton.classList.remove('btn-primary');
                addButton.classList.add('btn-secondary');
                addButton.disabled = true;
                addButton.textContent = artist.status;
            } else if (artist.status === "Failed to Add" || artist.status === "Invalid Path") {
                cardBody.classList.add('status-red');
                addButton.classList.remove('btn-primary');
                addButton.classList.add('btn-danger');
                addButton.disabled = true;
                addButton.textContent = artist.status;
            } else {
                cardBody.classList.add('status-blue');
                addButton.disabled = false;
            }
            return;
        }
    });
});

socket.on("lidarr_update", (response) => {
    lidarrTable.innerHTML = '';
    var allChecked = true;
    if (response.status == "busy") {
        getWantedLidarr.disabled = true;
        lidarrSpinner.classList.remove('d-none');
    }
    else {
        getWantedLidarr.disabled = false;
        lidarrSpinner.classList.add('d-none');
    }

    selectAllCheckbox.style.display = "block";
    selectAllCheckbox.checked = false;

    response.data.forEach((item, i) => {
        if (!item.checked) {
            allChecked = false;
        }
        var row = lidarrTable.insertRow();

        var cell1 = row.insertCell(0);
        var cell2 = row.insertCell(1);
        var cell3 = row.insertCell(2);

        var checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.className = "form-check-input";
        checkbox.id = "lidarr_" + i;
        checkbox.name = "lidarr_item";
        checkbox.checked = item.checked;
        checkbox.addEventListener("change", function () {
            checkIfAllTrue();
        });

        var label = document.createElement("label");
        label.className = "form-check-label";
        label.htmlFor = "lidarr_" + i;

        label.textContent = item.artist + " - " + item.album_name;

        cell1.appendChild(checkbox);
        cell2.appendChild(label);
        cell3.textContent = `${item.missing_count}/${item.track_count}`;
        cell3.classList.add("text-center");
    });
    selectAllCheckbox.checked = allChecked;
});

socket.once("music_recommendations", (response) => {

});

socket.on('music_recommendations', function (data) {
    const artistRow = document.getElementById('artist-row');
    artistRow.innerHTML = "";
    appendArtists(data.data);
    currentPage++;
    isLoading = false;
});