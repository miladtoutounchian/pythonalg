import pandas as pd
import urllib3
from keras.preprocessing import image
import pickle

# df = pd.read_excel('H:\My Stuff\mecent_pull.xlsx')
# df = df[pd.notnull(df['HAZARDMAPRATING'])]
# df = df[df['INSPECTIONTYPE'].str.contains('EL')]
# df1 = df[0:5]
# c = 0
# for i, j in zip(df['HAZARDMAPRATING'].values[:5], df['PHOTOLINK'].values[:5]):
#     m = j[7:].split('/')
#     # check if image is png
#     if m[4].split('.')[1] == 'png':
#         urlpath = 'http://dockerprodint01a.safetyauthority.ca' + '/' + m[3] + '/' + m[4]
#         http = urllib3.PoolManager()
#         r = http.request('GET', urlpath)
#         # check if we have the image url correctly
#         if r.status == 200:
#             with open("image.png", "wb") as code:
#                 code.write(r.data)
#             new_img = image.load_img('image.png', target_size=(224, 224))
#             x = image.img_to_array(new_img)
#             c += 1
#             print(x.flatten('F'))
#             # df1.assign(m=[[x.flatten('F')]])
#             df1['m'] = 0
#             df1.loc[c, 'm'] = [[x.flatten('F')]]
# df1.to_excel('sample.xlsx')
#print(df1)

df = pd.read_excel('H:\My Stuff\mecent_pull.xlsx')
df = df[pd.notnull(df['HAZARDMAPRATING'])]
df = df[df['INSPECTIONTYPE'].str.contains('EL')]
# print(df.head(20))
# for m in range(20):
#     print(df.loc[m, 'INSPECTIONTYPE'])
# print(df.loc[26, 'INSPECTIONTYPE'])
c = 0
s = []
for i, j in zip(df['HAZARDMAPRATING'].values[:2], df['PHOTOLINK'].values[:2]):
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
            new_img = image.load_img('image.png', target_size=(224, 224))
            x = image.img_to_array(new_img)
            file = open('image_as_text/image_{}.txt'.format(c), 'w')
            # with open('H:\My Stuff\image\image_{}.txt'.format(c), 'wb') as f:
            for element in x.flatten('F'):
                file.write(str(element) + '\n')

            # pickle.dump(x.flatten('F'), open('H:\My Stuff\image\image_{}.p'.format(c), 'wb'))
            s.append({'PHOTOLINK': "image_as_text/image_{}.txt".format(c)})
            c += 1
            # print(c)
            # print(df.loc[c, 'INSPECTIONTYPE'])

df.drop('INSPECTIONLINK', axis=1, inplace=True)
df1 = pd.DataFrame(s)
# # print(df1.shape)
df2 = df1.merge(df, on='PHOTOLINK', left_index=True, right_index=True)
# # print(df2.head())
df2.to_excel('sample2.xlsx', index=False)
# df3 = pd.read_excel('sample.xlsx')
# df3.head().to_excel('sample1.xlsx')
# print(len(df3['PHOTOLINK'].values[0]))
# for i in df3['PHOTOLINK'].values[0]:
#     print(i)

# y = pickle.load(open('H:\My Stuff\image\image_0.p', 'rb'))
# print(len(y))