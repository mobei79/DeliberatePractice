import os, requests, json

# BASE_URL = "http://127.0.0.1:5000"
BASE_URL = "http://10.10.204.24:5000"
ENTITY_URL = BASE_URL + '/query/inconsistency'

00

def test_inconsistency():

    id = ['+BDudTZ0V9YnLUnghhN6HrfTfXapO2332m+4e9ZSW+k=',  # 1
          '+DikbdOR2DImV83VvZEuCr55SlbI+7l6WDTiCLSSvDw=',  # 1
          '+qcylyi8zeOuhDinBo7x9aTVR9pgWs13KE8cFwKJ1HI=',  # 0
          '+radYtabqKKE6c7e6ceAd8A+Bp8scswwUdH/fZwwUFw='  # 0
          ]

    prop, delta = 'profession_station', 360

    data = {'id_card': id,
            'prop': prop,
            'income_tm_intervel': delta,
            'sim_method': 'equal',
            'percent': 0.9}


    r = requests.post(url=ENTITY_URL, data=json.dumps(data))
    print(r.json())
    assert r.status_code == 200


if __name__ == '__main__':
    # os.system('pytest -s test_app.py')
    # test_predict_entity()
    # test_predict_entity_batch()
    # test_async_predict_entity()
    test_inconsistency()