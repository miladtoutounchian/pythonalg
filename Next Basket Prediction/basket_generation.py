import json
import time
import datetime


def read_data(file_name):
    """Read a csv file that lists possible transactions"""
    result = list()
    with open(file_name, 'r') as file_reader:
        for line in file_reader:
            result.append(line.strip().split(';'))
    return result


def basket_gen(time_list, item_list):
    ind = [sorted(set(time_list)).index(v) for v in time_list]
    lst = [[] for _ in sorted(set(ind))]
    for i in range(len(item_list)):
        lst[ind[i]].append(item_list[i])
    return lst


def f(file_name):
    comp_list = read_data(file_name)
    all_customer_id = list()
    for i in range(len(comp_list)):
        all_customer_id.append(comp_list[i][1])
    unique_customer_id = list(set(all_customer_id))

    dictionary = list()
    basket_seq = list()
    basket_seq_short = list()
    for j in range(len(unique_customer_id)):
        list_item_by_customer = list()
        time_item_by_customer = list()
        for i in range(len(comp_list)):
            if comp_list[i][1] == unique_customer_id[j]:
                list_item_by_customer.append(comp_list[i][5])
                time_item_by_customer.append(comp_list[i][0])
        time_item_by_customer_timestamp = list()
        for k in range(len(time_item_by_customer)):
            time_item_by_customer_timestamp.append(
                time.mktime(datetime.datetime.strptime(time_item_by_customer[k], "%Y-%m-%d %H:%M:%S").timetuple()))
        basket_seq.append({unique_customer_id[j]: basket_gen(time_item_by_customer_timestamp, list_item_by_customer)})
        if len(basket_gen(time_item_by_customer, list_item_by_customer)) >= 10:
            basket_seq_short.append({unique_customer_id[j]: basket_gen(time_item_by_customer, list_item_by_customer)})
        # dictionary.append({unique_customer_id[j]:  list_item_by_customer, 'time': time_item_by_customer})
        dictionary.append({unique_customer_id[j]: list_item_by_customer})

    return unique_customer_id, dictionary, basket_seq, basket_seq_short


U, V, W, X = f('Tafeng.txt')

writer = open('tafeng_json_data', 'w')
for rec in W:
    writer.write(json.dumps(rec) + '\n')

writer = open('tafeng_json_data_more_transactions', 'w')
for rec in X:
    writer.write(json.dumps(rec) + '\n')


def data_set_property(file_name):
    unique_cust_id, cust_item_time_dict, basket_dict, basket_dict_large_trans = f(file_name)
    number_of_users = len(cust_item_time_dict)
    # number_of_items = len(list(set(sum(cust_item_time_dict.values(), []))))
    number_of_items = len(
        list(set(sum(sum([cust_item_time_dict[x].values() for x in range(len(cust_item_time_dict))], []), []))))
    # number_of_baskets = sum([len(x) for x in basket_dict.values()])
    list_of_basket = list()
    for y in range(len(basket_dict)):
        list_of_basket.append(sum([len(x) for x in basket_dict[y].values()]))
    number_of_baskets = sum(list_of_basket)
    return number_of_users, number_of_items, number_of_baskets
