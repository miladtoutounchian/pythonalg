import pandas as pd
import urllib3
import cv2
import os

df = pd.read_excel('H:\My Stuff\mecent_pull.xlsx')
df = df[pd.notnull(df['HAZARDMAPRATING'])]
df = df[df['INSPECTIONTYPE'].str.contains('EL')]
df_o = df
image_type = df_o['HAZARDMAPCELLREFERENCE'].unique()
# print(image_type)
# df = df[df['HAZARDMAPCELLREFERENCE'] == image_type[0]]
# print(df.head())
for i in range(len(image_type)):
    c = 0
    df = df_o[df_o['HAZARDMAPCELLREFERENCE'] == image_type[i]]
    print(image_type[i])
    for j in df['PHOTOLINK'].values:
        m = j[7:].split('/')
        # check if image is png
        if m[4].split('.')[1] == 'png':
            urlpath = 'http://dockerprodint01a.safetyauthority.ca' + '/' + m[3] + '/' + m[4]
            http = urllib3.PoolManager()
            r = http.request('GET', urlpath)
            # check if we have the image url correctly
            if r.status == 200:
                with open("image.png", "wb") as code:
                    code.write(r.data)

                if not os.path.exists('{}'.format(image_type[i])):
                    os.makedirs('{}'.format(image_type[i]))
                img = cv2.imread("image.png")
                cv2.imwrite('{}/{}.png'.format(image_type[i], c), img)
                c += 1

