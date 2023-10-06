//TMDB API
const my_api = '0826c0f9ab2e9bbfd6308fc84784e24f';
const BASE_URL = 'https://api.themoviedb.org/3';
const POP_URL = BASE_URL + '/discover/movie?sort_by=popularity.desc&api_key=' + my_api;
const TOP_URL = BASE_URL + '/discover/movie/?certification_country=US&certification=R&sort_by=vote_average.desc&api_key=' + my_api;
const KID_URL = BASE_URL + '/discover/movie?certification_country=US&certification.lte=G&sort_by=popularity.desc&api_key=' + my_api;
const IMG_URL = 'https://image.tmdb.org/t/p/w500';

const main = document.getElementById("main");
const main2 = document.getElementById("main2");
const main3 = document.getElementById("main3");

getMovies(POP_URL);
getMovies2(TOP_URL);
getMovies3(KID_URL);

function getMovies(url) {
    fetch(url).then(res => res.json()).then(data => {
        showMovies(data.results);
    })
}

function showMovies(data) {
    main.innerHTML = '';
    
    data.forEach(movie => {
        const {title, poster_path, vote_average, release_date, id} = movie;
        const movieEl = document.createElement('li');
        movieEl.innerHTML = `
        <div class="movie-card">
        <a href="#">
        <figure class="card-banner">
            <img src="${IMG_URL+poster_path}" alt="${title}">
            </figure>
            </a>
            <div class="title-wrapper">
            <a href="#">
            <h3 class="card-title">${title}</h3>
            </a>

            <time datetime=${release_date}>${release_date}</time>
            </div>

            <div class="card-meta">
                <div class="badge badge-outline">HD</div>

                <div class="duration">
            <ion-icon name="time-outline"></ion-icon>

            <time datetime="PT137M">${Math.floor((Math.random()*40)+100)} min</time>
            </div>

            <div class="rating">
            <ion-icon name="star"></ion-icon>

            <data>${vote_average}</data>
            </div>
            </div>
            </div>
        `
        main.appendChild(movieEl);
    })
}

function getMovies2(url) {
    fetch(url).then(res => res.json()).then(data => {
        console.log(data.results);
        showMovies2(data.results);
    })
}

function showMovies2(data) {
    main2.innerHTML = '';
    var i = 0;    
    data.forEach(movie => {
        if(i == 9){
            return;
        }
        i = i+1;
        if(i != 2) {
            const {title, poster_path, vote_average, release_date, id} = movie;
        const movieEl = document.createElement('li');
        movieEl.innerHTML = `
        <div class="movie-card">

        <a href="#">
          <figure class="card-banner">
            <img src="${IMG_URL+poster_path}" alt="${title}">
          </figure>
        </a>

        <div class="title-wrapper">
          <a href="#">
            <h3 class="card-title">${title}</h3>
          </a>

          <time datetime=${release_date}>${release_date}</time>
        </div>

        <div class="card-meta">
          <div class="badge badge-outline">2K</div>

          <div class="duration">
            <ion-icon name="time-outline"></ion-icon>

            <time datetime="PT122M">${Math.floor((Math.random()*40)+100)} min</time>
          </div>

          <div class="rating">
            <ion-icon name="star"></ion-icon>

            <data>${vote_average}</data>
          </div>
        </div>

      </div>
        `
        main2.appendChild(movieEl);
        }

    })
}

function getMovies3(url) {
    fetch(url).then(res => res.json()).then(data => {
        console.log(data.results);
        showMovies3(data.results);
    })
}

function showMovies3(data) {
    main3.innerHTML = '';
    i = 0;
    data.forEach(movie => {
        if(i == 8){
            return;
        }
        i = i+1;
        const {title, poster_path, vote_average, release_date, id} = movie;
        const movieEl = document.createElement('li');
        movieEl.innerHTML = `
        <div class="movie-card">

        <a href="#">
          <figure class="card-banner">
            <img src="${IMG_URL+poster_path}" alt="${title}">
          </figure>
        </a>

        <div class="title-wrapper">
          <a href="#">
            <h3 class="card-title">${title}</h3>
          </a>

          <time datetime=${release_date}>${release_date}</time>
        </div>

        <div class="card-meta">
          <div class="badge badge-outline">2K</div>

          <div class="duration">
            <ion-icon name="time-outline"></ion-icon>

            <time datetime="PT47M">${Math.floor((Math.random()*40)+100)} min</time>
          </div>

          <div class="rating">
            <ion-icon name="star"></ion-icon>

            <data>${vote_average}</data>
          </div>
        </div>

      </div>
        `
        main3.appendChild(movieEl);

    })
}