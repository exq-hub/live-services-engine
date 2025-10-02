from fastapi import APIRouter


##### Router
router = APIRouter()


@router.get("/hello")
def read_root():
    return {"message": "Hello"}
