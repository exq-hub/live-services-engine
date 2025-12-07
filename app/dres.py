import requests
from fastapi import APIRouter, Depends

from app.schemas import SubmitRequest
from app.utils import dump_log_msgpack, get_current_timestamp, get_shared_resources

##### Globals
dres_uri = "https://vbs.videobrowsing.org/api/v2"
dres_session = requests.Session()
dres_session.verify = True

login = {}
with open("dres.txt", "r") as f:
    info = [s.strip() for s in f.readlines()]
    login = {"username": info[0], "password": info[1]}
r = dres_session.post(dres_uri + "/login", json=login)
print(r.json())

##### Router
router = APIRouter()


@router.get("/dres/hello")
def hello(shared=Depends(get_shared_resources)):
    """Test connection"""
    log_message = {
        "timestamp": get_current_timestamp(),
        "action": "Initialize Session",
        "data": {"dres_session": dres_session},
    }

    # dump_log_msgpack(log_message, shared.logfile)

    return {"message": "dres is ready!", "sessionId": "FILL_ME_PLEASSSSSE"}


@router.post("/dres/submit")
def submit(body: SubmitRequest, shared=Depends(get_shared_resources)):
    """Submit media item or Q&A text to DRES server"""
    uri = f"{dres_uri}/submit/{body.evalId}"
    submission = {}
    start = 0
    end = 0
    collection = body.session_info.collection
    if body.qa:
        submission = {"answerSets": [{"answers": [{"text": body.text}]}]}
    elif body.itemId is not None:
        if (
            "segment_info"
            in shared.metadata[collection]["items"][body.itemId]["metadata"]
        ):
            start = (
                float(
                    shared.metadata[collection]["items"][body.itemId]["metadata"][
                        "segment_info"
                    ]["start"]
                )
                * 1000
            )
            end = (
                float(
                    shared.metadata[collection]["items"][body.itemId]["metadata"][
                        "segment_info"
                    ]["end"]
                )
                * 1000
            )

            midpoint_timestamp = start + (end - start) / 2

            name = shared.metadata[collection]["items"][body.itemId]["group"]
        submission = {
            "answerSets": [
                {
                    "answers": [
                        {
                            "mediaItemName": name,
                            "start": int(midpoint_timestamp),
                            "end": int(midpoint_timestamp),
                        }
                    ]
                }
            ]
        }
    elif body.start is not None:
        submission = {
            "answerSets": [
                {
                    "answers": [
                        {
                            "mediaItemName": body.name,
                            "start": int(float(body.start)),
                            "end": int(float(body.end)),
                        }
                    ]
                }
            ]
        }

    request_timestamp = get_current_timestamp()

    # Submit the data to the DRES server
    result = dres_session.post(uri, json=submission)
    print("DRES Response:", result.json())

    # Log the submission action
    request_log_message = {
        "timestamp": request_timestamp,
        "session": body.session_info.session,
        "action": "DRES Submit",
        "display_attrs": {
            "session": body.session_info.session,
            "modelId": body.session_info.modelId,
            "collection": body.session_info.collection,
            "name": body.name,
            "text": body.text,
            "qa": body.qa,
            "evalId": body.evalId,
            "submission": submission,
            "start": start,
            "end": end,
        },
    }
    # dump_log_msgpack(request_log_message, shared.logfile)

    # Log the result of the submission
    result_log_message = {
        "timestamp": get_current_timestamp(),
        "action": "DRES Response",
        "display_attrs": {"status_code": result.status_code, "response": result.json()},
    }
    # dump_log_msgpack(result_log_message, shared.logfile)

    return result.json()


@router.get("/dres/evaluation_list")
def get_active_evaluations():
    uri = f"{dres_uri}/evaluation/info/list"
    result = dres_session.get(uri)
    evaluations = [
        {"id": r["id"], "name": r["name"]}
        for r in result.json()
        if r["status"] == "ACTIVE"
    ]

    return evaluations
