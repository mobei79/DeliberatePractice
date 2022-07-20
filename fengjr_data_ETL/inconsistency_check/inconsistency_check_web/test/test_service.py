# coding:utf-8 -*- 
# @Time : 2020-02-13 10:57 
""" """

from inconsistency_check_web.service import query_diff

def test_query_diff():
    id = ['+BDudTZ0V9YnLUnghhN6HrfTfXapO2332m+4e9ZSW+k=', # 1
          '+DikbdOR2DImV83VvZEuCr55SlbI+7l6WDTiCLSSvDw=', # 1
          '+qcylyi8zeOuhDinBo7x9aTVR9pgWs13KE8cFwKJ1HI=', # 0
          '+radYtabqKKE6c7e6ceAd8A+Bp8scswwUdH/fZwwUFw='  # 0
        ]

    delta = 360
    props = 'profession_station'
    fn = ['equal','edit','longest','jaccard','jw']
    for f in fn:
        r = query_diff(id, props, delta, 0.9, sim_fn=f)
        print(f, r)

if __name__ == '__main__':
    test_query_diff()