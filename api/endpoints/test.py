from fastapi import APIRouter


router = APIRouter()

@router.get('/')
def test_service():
    return {'message': 'success'}