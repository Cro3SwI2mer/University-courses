import urllib
import datetime
import pandas as pd
import pandas_datareader.data as web

def RGBZCY(date):
    url = 'https://www.cbr.ru/hd_base/zcyc_params/zcyc/?DateTo='+date
    response = urllib.request.urlopen(url)
    myfile = response.read()
    file = open('output.txt', 'wb')
    file.write(myfile)
    file.close()
    with open('output.txt', 'br') as f:
        lines = list(f.readlines())[265:292]
    del lines[12:15]
    periods, yields = [], []
    periods = [float(str(lines[i]).replace("</th>\\r\\n'", "").replace("b'    <th>", "")) for i in range(12)]
    yields = [float(str(lines[i]).replace("</td>\\r\\n'", "").replace("b'    <td>", "")) for i in range(12, 24)]
    table = pd.DataFrame({'Period': periods, 'Yield': yields}).set_index('Period')
    return table

def MOEX_data(ticker, start_date, end_date):
    # ticker = 'IMOEX'
    df = web.DataReader(ticker, 'moex', start_date, end_date)
    df = df[['CLOSE']]
    r = [((df['CLOSE'][i]-df['CLOSE'][i-1])/df['CLOSE'][i-1])-1 for i in range(1, len(df))]
    r.insert(0, 0.0)
    df['Return'] = r
    return df

print(MOEX_data('MGNT', '2018-11-28', '2019-11-29'))
