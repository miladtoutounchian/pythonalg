import math
from itertools import chain, izip


def SlidingWindow(window,tokens):

	# Stores the tokens for the text that is input by the user
	mylist = tokens

	window_val =[]

	j = 1
	k = 2 
	p = 0
	w = []

	for i in mylist:
		if len(mylist[:-(len(mylist)-mylist.index(i))]) <= (window/2):
			previous = mylist[:-(len(mylist)-mylist.index(i))]
		else:
			previous = mylist[j:-(len(mylist)-mylist.index(i))]	
			j += 1
		
		current = mylist[mylist.index(i)]
		
		prev_len = len(previous)
		
		if prev_len == (window/2):
			nxt = mylist[mylist.index(i)+1:mylist.index(i)+(window/2)+1]
		else:
			nxt = mylist[mylist.index(i)+1:mylist.index(i)+(window-prev_len)]


		if len(nxt) <= window/3:
			previous = mylist[j-k+p:-(len(mylist)-mylist.index(i))]	
			p -= 1

		#print previous, current, nxt
		w.append([previous, current, nxt])

	return w
