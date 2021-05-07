from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder

from movies_database.movies_data import MovieDB, TMDBApi
from movies_recommender.simple_recomm import simple_recomm
from movies_recommender.content_based import content_recomm1, content_recomm2
from movies_recommender.collaborative_based import collaborative_recomm

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


""" SHARED RESOURCES """
resources = {}


@app.on_event("startup")
async def startup_event():
    resources["movie_db"] = MovieDB()
    resources["tmdb_api"] = TMDBApi()
    resources["user_ratings"] = []
    resources["similar_movies"]=[]
    resources["recommended_movies"]=[]



@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("landing_page.html", {"request": request})



@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    movies_list = simple_recomm()[0][:3]
    movies_dict = [{"movie": title} for title in movies_list]

    movie_overview3=simple_recomm()[1][:3]

    movie_ids = simple_recomm()[2][:3]
    movies_poster_urls = [
        resources["tmdb_api"].get_first_movie_poster_url(mid) for mid in movie_ids
    ]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "movies3": movies_dict,
            "movie_overview3":movie_overview3,
            "posters_urls": movies_poster_urls,
        },
    )




@app.get("/top-movies", response_class=HTMLResponse)
async def top_movies(request: Request):
    movies_list = simple_recomm()[0]
    movies_overview=simple_recomm()[1]

    movie_ids = simple_recomm()[2]
    movies_poster_urls_top10 = [
        resources["tmdb_api"].get_first_movie_poster_url(mid) for mid in movie_ids
    ]

    return templates.TemplateResponse(
        "movies_top.html",
        {
            "request": request,
            "movies_titles": movies_list,
            "movies_overview": movies_overview,
            "posters_urls_top10": movies_poster_urls_top10,
        },
    )




## find me a movie

@app.get("/find-me-movie", response_class=HTMLResponse)
async def find_me_movie(request: Request):
    return templates.TemplateResponse(
        "find_me_movie.html",
        {
            "request": request, 
            "user_ratings": resources["user_ratings"], 
            "similar_movies": resources["similar_movies"],
            "recommended_movies": resources["recommended_movies"]
        }, 
    )



@app.post("/ratings/forms/add")
async def add_rating(
    response_class: RedirectResponse,
    title: str = Form(...),
    rating: int = Form(...),
):
    resources["user_ratings"].append([resources["movie_db"].get_movieID(title), title, rating])

    return RedirectResponse("/find-me-movie", status_code=303)



@app.post("/ratings/forms/delete")
async def remove_rating(
    response_class: RedirectResponse,
    id: str = Form(...),
):
    resources["user_ratings"] = [x for x in resources["user_ratings"] if x[0] != id]

    return RedirectResponse("/find-me-movie", status_code=303)



@app.post('/forms/contentrecomm2', response_class=HTMLResponse)
def contentrecomm2 (
    request: Request,
    title: str = Form(...)
    ):
    if title==None:
        return RedirectResponse("/find-me-movie", status_code=303)
    resources["similar_movies"]=content_recomm2(title)
    return RedirectResponse("/find-me-movie", status_code=303)



@app.post('/forms/collaborativerecomm',response_class=HTMLResponse)
def collaborativerecomm(
    request: Request
):
    titles_liste=[x[1] for x in resources["user_ratings"]]
    values_liste=[x[2] for x in resources["user_ratings"]]
    resources["recommended_movies"]=collaborative_recomm(titles_liste, values_liste)
    return RedirectResponse("/find-me-movie", status_code=303)




#####


@app.post('/simplerecomm/')
def simplerecomm ():
    return(simple_recomm())


@app.post('/contentrecomm1/{title}')
def contentrecomm1 (title:str):
    return(content_recomm1(title))


@app.post('/contentrecomm2/{title}')
def contentrecomm2 (title:str):
    return(content_recomm2(title))

@app.post('/collaborativerecomm/{titles,values}')
def collaborativerecomm(titles:list, values: list):
    return (collaborative_recomm(titles, values))

