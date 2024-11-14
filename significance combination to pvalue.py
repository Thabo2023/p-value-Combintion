#!/usr/bin/env python
# coding: utf-8

# # BY THABO PILUSA

# In[1]:


import numpy as np
import csv
from scipy.stats import norm
from scipy.stats import chi2


# # 1: ATLAS COMBINATION

# 1.1 acceptances at different alpha for different masses( 3000 GeV to 4000 GeV)

# In[2]:


#M = 3000
w_h3k = 0.00316 #alpha = 25
w_i3k = 0.00316 #alpha = 27
w_j3k = 0.01366 #alpha = 29
w_k3k = 0.0819 #alpha = 31
w_l3k = 0.00624  #alpha = 33


# In[3]:


#M = 3100
w_h31k = 0.00418 #alpha = 25
w_i31k = 0.00418 #alpha = 27
w_j31k = 0.03852 #alpha = 29
w_k31k = 0.07344 #alpha = 31
w_l31k = 0.00156 #alpha = 33


# In[4]:


#M = 3200
w_h32k = 0.00504  #alpha = 25
w_i32k = 0.04198 #alpha = 27
w_j32k = 0.04198  #alpha = 29
w_k32k = 0.0852 #alpha = 31
w_l32k = 0.00334#alpha = 33


# In[5]:


#M = 3300
w_h33k = 0.01066 #alpha = 25
w_i33k = 0.09438 #alpha = 27
w_j33k = 0.09438 #alpha = 29
w_k33k = 0.04074 #alpha = 31
w_l33k = 0.0012  #alpha = 33


# In[6]:


#M = 3400
w_h34k = 0.00718 #alpha = 25
w_i34k = 0.06156 #alpha = 27
w_j34k = 0.09232  #alpha = 29
w_k34k = 0.0027#alpha = 31
w_l34k = 0.00062 #alpha = 33


# In[7]:


#M = 3500
w_h35k = 0.01646 #alpha = 25
w_i35k = 0.1222 #alpha = 27
w_j35k = 0.04092 #alpha = 29
w_k35k = 0.00138#alpha = 31
w_l35k = 0.00064 #alpha = 33


# In[8]:


#M = 3600
w_h36k = 0.03502 #alpha = 25
w_i36k = 0.1437 #alpha = 27
w_j36k = 0.01094#alpha = 29
w_k36k = 0.00072#alpha = 31
w_l36k = 0.00036#alpha = 33


# In[9]:


#M = 3700
w_h37k = 0.13208 #alpha = 25
w_i37k = 0.05086 #alpha = 27
w_j37k = 0.0011 #alpha = 29
w_k37k = 0.00028 #alpha = 31
w_l37k = 0.00038 #alpha = 33


# In[10]:


#M = 3800
w_h38k = 0.1601#alpha = 25
w_i38k = 0.01714 #alpha = 27
w_j38k =0.00074 #alpha = 29
w_k38k =0.00038 #alpha = 31
w_l38k = 0.00028 #alpha = 33


# In[11]:


#M = 3900
w_h39k = 0.13132 #alpha = 25
w_i39k = 0.00484#alpha = 27
w_j39k = 0.00036 #alpha = 29
w_k39k = 0.00034 #alpha = 31
w_l39k = 0.00034 #alpha = 33


# In[12]:


#M = 4000
w_h4k =  0.07636 #alpha = 25
w_i4k = 0.0016 #alpha = 27
w_j4k = 0.00044 #alpha = 29
w_k4k = 0.00028#alpha = 31
w_l4k = 0.00018 #alpha = 33


# 1.2 Take Significance along with the weight/acceptance then combine it to p-value

# In[13]:


M = [3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000] 


#Significance ATLAS

sigh = [0,1.5,2.102,2,0.88,0,0,0,0,0,0.478] #alpha = 0.25

sigi = [2.1,2.1,2.1,0.3,0.3,0,0,0,0,0,0] #alpha = 0.27

sigj = [0,0,1.45,2.03,2.45,1.4,0.55,0,0,0,0] #alpha = 0.29

sigk = [1.1,1.105,0.88,0.1,0.05,0,0,0,0,0,0] #alpha = 0.31
   

sigl = [1.12,1.5,1.21,0.2,0,0,0,0,0.01,0.5,0.55] #alpha = 0.33
    

#These are the acceptances 
ah = [w_h3k,w_h31k,w_h32k,w_h33k,w_h34k,w_h35k,w_h36k,w_h37k,w_h38k,w_h39k,w_h4k] # acceptance at alpha = 0.25
ai = [w_i3k,w_i31k,w_i32k,w_i33k,w_i34k,w_i35k,w_i36k,w_i37k,w_i38k,w_i39k,w_i4k] # acceptance at alpha = 0.27
aj = [w_j3k,w_j31k,w_j32k,w_j33k,w_j34k,w_j35k,w_j36k,w_j37k,w_j38k,w_j39k,w_j4k] # acceptance at alpha = 0.29
ak = [w_k3k,w_k31k,w_k32k,w_k33k,w_k34k,w_k35k,w_k36k,w_k37k,w_k38k,w_k39k,w_k4k] # acceptance at alpha = 0.31
al = [w_l3k,w_l31k,w_l32k,w_l33k,w_l34k,w_l35k,w_l36k,w_l37k,w_l38k,w_l39k,w_l4k] # acceptance at alpha = 0.33


# x_up and x_down defines the function inside the normal cdf
x_i = [ai[0]*sigi[0], ai[1]*sigi[1], ai[2]*sigi[2], ai[3]*sigi[3], ai[4]*sigi[4], ai[5]*sigi[5], ai[6]*sigi[6], ai[7]*sigi[7], ai[8]*sigi[8], ai[9]*sigi[9], ai[10]*sigi[10]]
x_j = [aj[0]*sigj[0], aj[1]*sigj[1], aj[2]*sigj[2], aj[3]*sigj[3], aj[4]*sigj[4], aj[5]*sigj[5], aj[6]*sigj[6], aj[7]*sigj[7], aj[8]*sigj[8], aj[9]*sigj[9], aj[10]*sigj[10]]
x_k = [ak[0]*sigk[0], ak[1]*sigk[1], ak[2]*sigk[2], ak[3]*sigk[3], ak[4]*sigk[4], ak[5]*sigk[5], ak[6]*sigk[6], ak[7]*sigk[7], ak[8]*sigk[8], ak[9]*sigk[9], ak[10]*sigk[10]]
x_l = [al[0]*sigl[0], al[1]*sigl[1], al[2]*sigl[2], al[3]*sigl[3], al[4]*sigl[4], al[5]*sigl[5], al[6]*sigl[6], al[7]*sigl[7], al[8]*sigl[8], al[9]*sigl[9], al[10]*sigl[10]]


x_upw = [(x_i[0] + x_j[0] + x_k[0] + x_l[0]), (x_i[1] + x_j[1] + x_k[1] + x_l[1]), (x_i[2] + x_j[2] + x_k[2] + x_l[2]), (x_i[3] + x_j[3] + x_k[3] + x_l[3]), (x_i[4] + x_j[4] + x_k[4] + x_l[4]), (x_i[5] + x_j[5] + x_k[5] + x_l[5]), (x_i[6] + x_j[6] + x_k[6] + x_l[6]), (x_i[7] + x_j[7] + x_k[7] + x_l[7]), (x_i[8] + x_j[8] + x_k[8] + x_l[8]), (x_i[9] + x_j[9] + x_k[9] + x_l[9]), (x_i[10] + x_j[10] + x_k[10] + x_l[10])]

x_dow =[np.sqrt(ai[0]**2 + aj[0]**2 + ak[0]**2 + al[0]**2), np.sqrt(ai[1]**2 + aj[1]**2 + ak[1]**2 + al[1]**2), np.sqrt(ai[2]**2 + aj[2]**2 + ak[2]**2 + al[2]**2), np.sqrt(ai[3]**2 + aj[3]**2 + ak[3]**2 + al[3]**2), np.sqrt(ai[4]**2 + aj[4]**2 + ak[4]**2 + al[4]**2), np.sqrt(ai[5]**2 + aj[5]**2 + ak[5]**2 + al[5]**2), np.sqrt(ai[6]**2 + aj[6]**2 + ak[6]**2 + al[6]**2), np.sqrt(ai[7]**2 + aj[7]**2 + ak[7]**2 + al[7]**2), np.sqrt(ai[8]**2 + aj[8]**2 + ak[8]**2 + al[8]**2), np.sqrt(ai[9]**2 + aj[9]**2 + ak[9]**2 + al[9]**2), np.sqrt(ai[10]**2 + aj[10]**2 + ak[10]**2 + al[10]**2)]


x = [x_upw[0]/x_dow[0],x_upw[1]/x_dow[1], x_upw[2]/x_dow[2], x_upw[3]/x_dow[3], x_upw[4]/x_dow[4], x_upw[5]/x_dow[5], x_upw[6]/x_dow[6], x_upw[7]/x_dow[7], x_upw[8]/x_dow[8], x_upw[9]/x_dow[9], x_upw[10]/x_dow[10]]  



pvalue = [2*(1 - norm.cdf(x[0])), 2*(1 - norm.cdf(x[1])), 2*(1 - norm.cdf(x[2])),2*(1 - norm.cdf(x[3])), 2*(1 - norm.cdf(x[4])), 2*(1 - norm.cdf(x[5])), 2*(1 - norm.cdf(x[6])), 2*(1 - norm.cdf(x[7])), 2*(1 - norm.cdf(x[8])), 2*(1 - norm.cdf(x[9])), 2*(1 - norm.cdf(x[10]))]          


file = zip(M, pvalue)
with open('thecombz.txt', 'w') as combination:
    for (M, pvalue) in file:
      combination.write("{0} {1}\n".format(M, pvalue))
print('File created')
 
        
         


# # 2 : convert ATLAS significance to p-value

# In[63]:


#For alpha = 25
M = [3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000] 

pvalueh = [ 2*(1 - norm.cdf(sigh[0])), 2*(1 - norm.cdf(sigh[1])), 2*(1 - norm.cdf(sigh[2])), 2*(1 - norm.cdf(sigh[3])), 2*(1 - norm.cdf(sigh[4])), 2*(1 - norm.cdf(sigh[5])),2*(1 - norm.cdf(sigh[6])),2*(1 - norm.cdf(sigh[7])), 2*(1 - norm.cdf(sigh[8])), 2*(1 - norm.cdf(sigh[9])), 2*(1 - norm.cdf(sigh[10]))]


file = zip(M, pvalueh)
with open('pvalueh.txt', 'w') as combination:
    for (M, pvalueh) in file:
      combination.write("{0} {1}\n".format(M, pvalueh))
print('File created')


# In[64]:


#For alpha = 27
M = [3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000]

pvaluei = [2*(1 - norm.cdf(sigi[0])),2*(1 - norm.cdf(sigi[1])), 2*(1 - norm.cdf(sigi[2])), 2*(1 - norm.cdf(sigi[3])),2*(1 - norm.cdf(sigi[4])),2*(1 - norm.cdf(sigi[5])), 2*(1 - norm.cdf(sigi[6])), 2*(1 - norm.cdf(sigi[7])), 2*(1 - norm.cdf(sigi[8])),2*(1 - norm.cdf(sigi[9])), 2*(1 - norm.cdf(sigi[10]))]

file = zip(M, pvaluei)
with open('pvaluei.txt', 'w') as combination:
    for (M, pvaluei) in file:
      combination.write("{0} {1}\n".format(M, pvaluei))
print('File created')


# In[65]:


#For alpha = 29
M = [3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000]

pvaluej = [2*(1 - norm.cdf(sigj[0])), 2*(1 - norm.cdf(sigj[1])), 2*(1 - norm.cdf(sigj[2])), 2*(1 - norm.cdf(sigj[3])),2*(1 - norm.cdf(sigj[4])), 2*(1 - norm.cdf(sigj[5])), 2*(1 - norm.cdf(sigj[6])), 2*(1 - norm.cdf(sigj[7])), 2*(1 - norm.cdf(sigj[8])), 2*(1 - norm.cdf(sigj[9])), 2*(1 - norm.cdf(sigj[10]))]


file = zip(M, pvaluej)
with open('pvaluej.txt', 'w') as combination:
    for (M, pvaluej) in file:
      combination.write("{0} {1}\n".format(M, pvaluej))
print('File created')
           


# In[66]:


#For alpha = 31
M = [3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000]

pvaluek = [2*(1 - norm.cdf(sigk[0])), 2*(1 - norm.cdf(sigk[1])), 2*(1 - norm.cdf(sigk[2])), 2*(1 - norm.cdf(sigk[3])), 2*(1 - norm.cdf(sigk[4])), 2*(1 - norm.cdf(sigk[5])),2*(1 - norm.cdf(sigk[6])),2*(1 - norm.cdf(sigk[7])),2*(1 - norm.cdf(sigk[8])),2*(1 - norm.cdf(sigk[9])),2*(1 - norm.cdf(sigk[10]))]

file = zip(M, pvaluek)
with open('pvaluek.txt', 'w') as combination:
    for (M, pvaluek) in file:
      combination.write("{0} {1}\n".format(M, pvaluek))
print('File created')



# In[67]:


#For alpha = 33
M = [3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000]
pvaluel = [2*(1 - norm.cdf(sigl[0])), 2*(1 - norm.cdf(sigl[1])), 2*(1 - norm.cdf(sigl[2])), 2*(1 - norm.cdf(sigl[3])), 2*(1 - norm.cdf(sigl[4])), 2*(1 - norm.cdf(sigl[5])), 2*(1 - norm.cdf(sigl[6])), 2*(1 - norm.cdf(sigl[7])), 2*(1 - norm.cdf(sigl[8])), 2*(1 - norm.cdf(sigl[9])), 2*(1 - norm.cdf(sigl[10]))]


file = zip(M, pvaluel)
with open('pvaluel.txt', 'w') as combination:
    for (M, pvaluel) in file:
      combination.write("{0} {1}\n".format(M, pvaluel))
print('File created')
         


# # ATLAS SIGNIFICANCE USING FISHER METHOD

# In[71]:


M = [3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000]
pvaluei = [0.035728841125633126,0.035728841125633126,0.035728841125633126,0.7641771556220949,0.7641771556220949,1.0,1.0,1.0,1.0,1.0,1.0]
pvaluej = [1.0,1.0,0.1470585192192968,0.04235653928534444,0.014285621470542909,0.16151331846754213,0.5823193735766927,1.0,1.0,1.0,1.0]
pvaluek = [0.27133212189276534,0.2691595891287921,0.3788593095534243,0.920344325445942,0.960122388323255,1.0,1.0,1.0,1.0,1.0,1.0]
pvaluel = [0.2627137620854614,0.13361440253771617,0.22627889288795489,0.841480581121794,1.0,1.0,1.0,1.0,0.9920212873707368,0.6170750774519738,0.5823193735766927]


chisquaredATLAS = [-2*(np.log(pvaluei[0]) + np.log(pvaluej[0]) + np.log(pvaluek[0]) + np.log(pvaluel[0])), -2*(np.log(pvaluei[1]) + np.log(pvaluej[1]) + np.log(pvaluek[1]) + np.log(pvaluel[1])), -2*(np.log(pvaluei[2]) + np.log(pvaluej[2]) + np.log(pvaluek[2]) + np.log(pvaluel[2])), -2*(np.log(pvaluei[3]) + np.log(pvaluej[3]) + np.log(pvaluek[3]) + np.log(pvaluel[3])), -2*(np.log(pvaluei[4]) + np.log(pvaluej[4]) + np.log(pvaluek[4]) + np.log(pvaluel[4])), -2*(np.log(pvaluei[5]) + np.log(pvaluej[5]) + np.log(pvaluek[5]) + np.log(pvaluel[5])), -2*(np.log(pvaluei[6]) + np.log(pvaluej[6]) + np.log(pvaluek[6]) + np.log(pvaluel[6])), -2*(np.log(pvaluei[7]) + np.log(pvaluej[7]) + np.log(pvaluek[7]) + np.log(pvaluel[7])), -2*(np.log(pvaluei[8]) + np.log(pvaluej[8]) + np.log(pvaluek[8]) + np.log(pvaluel[8])), -2*(np.log(pvaluei[9]) + np.log(pvaluej[9]) + np.log(pvaluek[9]) + np.log(pvaluel[9])), -2*(np.log(pvaluei[10]) + np.log(pvaluej[10]) + np.log(pvaluek[10]) + np.log(pvaluel[10]))]
tests = 4
dof = 2*tests  
ATLASpvalue = [1 - chi2.cdf(chisquaredATLAS[0], dof), 1 - chi2.cdf(chisquaredATLAS[1], dof), 1 - chi2.cdf(chisquaredATLAS[2], dof), 1 - chi2.cdf(chisquaredATLAS[3], dof), 1 - chi2.cdf(chisquaredATLAS[4], dof), 1 - chi2.cdf(chisquaredATLAS[5], dof), 1 - chi2.cdf(chisquaredATLAS[6], dof), 1 - chi2.cdf(chisquaredATLAS[7], dof), 1 - chi2.cdf(chisquaredATLAS[8], dof), 1 - chi2.cdf(chisquaredATLAS[9], dof), 1 - chi2.cdf(chisquaredATLAS[10], dof)] 



file = zip(M, ATLASpvalue)
with open('FisherATLAS.txt', 'w') as combination:
    for (M,ATLASpvalue) in file:
      combination.write("{0} {1}\n".format(M, ATLASpvalue))
print('File created')

    
    
    


# # 3: ATLAS and CMS Combnation using Fisher's Method

# In[119]:


M = [3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000]

atlaspval = [0.21324369664620635,0.26655900034494406,0.028170608716268752,0.10813275009149392,0.027429417873764406,0.6566720265542889,0.9666976068511732,1.0,0.9998698277003157,0.9721941172018898,0.9533543862141889]

cmspval =[0.09957249133816881,0.021057734028820718,0.009777135037203122,0.005418798909150091,0.001520994605028525,0.0007661510179304276,4.082808504768387e-05,0.0029793642644895257,0.01523175616577399,0.0082429074825352,0.008182641430289017]





chisquared = [-2*(np.log(atlaspval[0]) + np.log(cmspval[0])), -2*(np.log(atlaspval[1]) + np.log(cmspval[1])), -2*(np.log(atlaspval[2]) + np.log(cmspval[2])), -2*(np.log(atlaspval[3]) + np.log(cmspval[3])), -2*(np.log(atlaspval[4]) + np.log(cmspval[4])), -2*(np.log(atlaspval[5]) + np.log(cmspval[5])), -2*(np.log(atlaspval[6]) + np.log(cmspval[6])), -2*(np.log(atlaspval[7]) + np.log(cmspval[7])), -2*(np.log(atlaspval[8]) + np.log(cmspval[8])), -2*(np.log(atlaspval[9]) + np.log(cmspval[9])), -2*(np.log(atlaspval[10]) + np.log(cmspval[10]))]

k = 2       # 2 independent tests
df = 2*k  # Degrees of freedom

ATLASCMSpvalue = [1 - chi2.cdf(chisquared[0], df), 1 - chi2.cdf(chisquared[1], df), 1 - chi2.cdf(chisquared[2], df), 1 - chi2.cdf(chisquared[3], df), 1 - chi2.cdf(chisquared[4], df), 1 - chi2.cdf(chisquared[5], df), 1 - chi2.cdf(chisquared[6], df), 1 - chi2.cdf(chisquared[7], df), 1 - chi2.cdf(chisquared[8], df), 1 - chi2.cdf(chisquared[9], df), 1 - chi2.cdf(chisquared[10], df)]


file = zip(M, ATLASCMSpvalue)
with open('ATLASCMSpvalue.txt', 'w') as combination:
    for (M,ATLASCMSpvalue) in file:
      combination.write("{0} {1}\n".format(M, ATLASCMSpvalue))
print('File created')







# # ATLAS & CMS p-value combination using the same weight(acceptance) of 0.5

# In[162]:


M = [3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000]


atlaspval = [0.21324369664620635,0.26655900034494406,0.028170608716268752,0.10813275009149392,0.027429417873764406,0.6566720265542889,0.9666976068511732,1.0,0.9998698277003157,0.9721941172018898,0.9533543862141889]

cmspval =[0.09957249133816881,0.021057734028820718,0.009777135037203122,0.005418798909150091,0.001520994605028525,0.0007661510179304276,4.082808504768387e-05,0.0029793642644895257,0.01523175616577399,0.0082429074825352,0.008182641430289017]


atlassig = [norm.ppf(1 - atlaspval[0]/ 2),norm.ppf(1 - atlaspval[1]/ 2),norm.ppf(1 - atlaspval[2]/ 2),norm.ppf(1 - atlaspval[3]/ 2),norm.ppf(1 - atlaspval[4]/ 2),norm.ppf(1 - atlaspval[5]/ 2),norm.ppf(1 - atlaspval[6]/ 2),norm.ppf(1 - atlaspval[7]/ 2),norm.ppf(1 - atlaspval[8]/ 2),norm.ppf(1 - atlaspval[9]/ 2),norm.ppf(1 - atlaspval[10]/ 2)]
cmssig   = [norm.ppf(1 - cmspval[0]/ 2),norm.ppf(1 - cmspval[1]/ 2),norm.ppf(1 - cmspval[2]/ 2),norm.ppf(1 - cmspval[3]/ 2),norm.ppf(1 - cmspval[4]/ 2),norm.ppf(1 - cmspval[5]/ 2),norm.ppf(1 - cmspval[6]/ 2),norm.ppf(1 - cmspval[7]/ 2),norm.ppf(1 - cmspval[8]/ 2),norm.ppf(1 - cmspval[9]/ 2),norm.ppf(1 - cmspval[10]/ 2)]

w = 0.5 #weight

xcomb = [w*atlassig[0] + w*cmssig[0],w*atlassig[1] + w*cmssig[1], w*atlassig[2] + w*cmssig[2], w*atlassig[3] + w*cmssig[3], w*atlassig[4] + w*cmssig[4], w*atlassig[5] + w*cmssig[5], w*atlassig[6] + w*cmssig[6], w*atlassig[7] + w*cmssig[7], w*atlassig[8] + w*cmssig[8], w*atlassig[9] + w*cmssig[9], w*atlassig[10] + w*cmssig[10]]

combpval = [2*(1 - norm.cdf(xcomb[0])), 2*(1 - norm.cdf(xcomb[1])), 2*(1 - norm.cdf(xcomb[2])), 2*(1 - norm.cdf(xcomb[3])), 2*(1 - norm.cdf(xcomb[4])), 2*(1 - norm.cdf(xcomb[5])), 2*(1 - norm.cdf(xcomb[6])), 2*(1 - norm.cdf(xcomb[7])), 2*(1 - norm.cdf(xcomb[8])), 2*(1 - norm.cdf(xcomb[9])), 2*(1 - norm.cdf(xcomb[10]))]


file = zip(M, combpval)
with open('combinationpval.txt', 'w') as combination:
    for (M,combpval) in file:
      combination.write("{0} {1}\n".format(M, combpval))
print('File created')





# In[3]:


thabo = norm.ppf(1 - (2.020528189206061e-11/ 2))
print(thabo)


# In[ ]:




