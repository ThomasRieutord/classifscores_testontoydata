#!/usr/bin/env python3
# coding: utf-8

# Tests des scores de classifications internes sur des données-jouets
# ===============


import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs,make_moons
from sklearn.metrics import silhouette_score,calinski_harabaz_score,davies_bouldin_score


N_values= [200,500,1000,2000,5000,10000]
goodclassif_seed=418
badclassif_seed=4

S_score_gblobs = []
CH_score_gblobs = []
DB_score_gblobs = []
S_chrono_gblobs = []
CH_chrono_gblobs = []
DB_chrono_gblobs = []

S_score_bblobs = []
CH_score_bblobs = []
DB_score_bblobs = []
S_chrono_bblobs = []
CH_chrono_bblobs = []
DB_chrono_bblobs = []

S_score_moons = []
CH_score_moons = []
DB_score_moons = []
S_chrono_moons = []
CH_chrono_moons = []
DB_chrono_moons = []

for N in N_values:
	
	# Good blobs
	#------------
	X, y = make_blobs(n_samples=N,random_state=goodclassif_seed)
	
	t0=time.time()
	S_score_gblobs.append(silhouette_score(X,y))
	t1=time.time()
	S_chrono_gblobs.append(t1-t0)
	
	CH_score_gblobs.append(calinski_harabaz_score(X,y))
	t2=time.time()
	CH_chrono_gblobs.append(t2-t1)
	
	DB_score_gblobs.append(davies_bouldin_score(X,y))
	t3=time.time()
	DB_chrono_gblobs.append(t3-t2)
	
	
	# Bad blobs
	#------------
	X, y = make_blobs(n_samples=N,random_state=badclassif_seed)
	
	t0=time.time()
	S_score_bblobs.append(silhouette_score(X,y))
	t1=time.time()
	S_chrono_bblobs.append(t1-t0)
	
	CH_score_bblobs.append(calinski_harabaz_score(X,y))
	t2=time.time()
	CH_chrono_bblobs.append(t2-t1)
	
	DB_score_bblobs.append(davies_bouldin_score(X,y))
	t3=time.time()
	DB_chrono_bblobs.append(t3-t2)
	
	
	# Moons
	#-------
	X, y = make_moons(n_samples=N,noise=0.01)
	
	t0=time.time()
	S_score_moons.append(silhouette_score(X,y))
	t1=time.time()
	S_chrono_moons.append(t1-t0)
	
	CH_score_moons.append(calinski_harabaz_score(X,y))
	t2=time.time()
	CH_chrono_moons.append(t2-t1)
	
	DB_score_moons.append(davies_bouldin_score(X,y))
	t3=time.time()
	DB_chrono_moons.append(t3-t2)

plt.figure()
plt.title("Silhouette score evolution with $N$")
plt.plot(N_values,S_score_gblobs,'b-o',label="Good blobs")
plt.plot(N_values,S_score_bblobs,'b-x',label="Bad blobs")
plt.plot(N_values,S_score_moons,'b-*',label="Moons")
plt.grid()
plt.legend()
plt.xlabel("Number of points")
plt.ylabel("Score value")
plt.show(block=False)

plt.figure()
plt.title("Calinski-Harabasz score evolution with $N$")
plt.plot(N_values,CH_score_gblobs,'r-o',label="Good blobs")
plt.plot(N_values,CH_score_bblobs,'r-x',label="Bad blobs")
plt.plot(N_values,CH_score_moons,'r-*',label="Moons")
plt.grid()
plt.legend()
plt.xlabel("Number of points")
plt.ylabel("Score value")
plt.show(block=False)

plt.figure()
plt.title("Davies-Bouldin score evolution with $N$")
plt.plot(N_values,DB_score_gblobs,'g-o',label="Good blobs")
plt.plot(N_values,DB_score_bblobs,'g-x',label="Bad blobs")
plt.plot(N_values,DB_score_moons,'g-*',label="Moons")
plt.grid()
plt.legend()
plt.xlabel("Number of points")
plt.ylabel("Score value")
plt.show(block=False)



plt.figure()
plt.title("Calculation time")
plt.plot(N_values,S_chrono_gblobs,'b:o',label="Silhouette")
plt.plot(N_values,CH_chrono_gblobs,'r:o',label="Calinski-Harabasz")
plt.plot(N_values,DB_chrono_gblobs,'g:o',label="Davies-Bouldin")
plt.plot(N_values,S_chrono_bblobs,'b:x')
plt.plot(N_values,CH_chrono_bblobs,'r:x')
plt.plot(N_values,DB_chrono_bblobs,'g:x')
plt.plot(N_values,S_chrono_moons,'b:*')
plt.plot(N_values,CH_chrono_moons,'r:*')
plt.plot(N_values,DB_chrono_moons,'g:*')
plt.grid()
plt.legend()
plt.xlabel("Number of points")
plt.ylabel("Seconds")
plt.show(block=False)

input("Press enter to finish")


