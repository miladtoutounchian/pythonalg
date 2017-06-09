import json
from sklearn import preprocessing

data = list()
with open('tafeng_json_data_more_transactions') as f:
    for line in f:
        data.append(sum(list(json.loads(line).values()[0]), []))

all_item_id = sum(data, [])

print(len(list(set(all_item_id))))
# print()

unique_item_id = list(set(all_item_id))
le = preprocessing.LabelEncoder()
le.fit(unique_item_id)

cmp_data = list()
with open('tafeng_json_data_more_transactions') as f:
    for line in f:
        A = list()
        for item_id in json.loads(line).values()[0]:
            A.append(le.transform(item_id).tolist())
        cmp_data.append({json.loads(line).keys()[0]: A})
# print(cmp_data)

writer = open('tafeng_json_data_more_transactions_item_id', 'w')
for rec in cmp_data:
    writer.write(json.dumps(rec) + '\n')
writer.close()