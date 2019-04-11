import numpy as np 
import matplotlib.pyplot as plt 

#Variables 
TransitionMatrix = np.array([
[0.000, 0.000, 0.000, 0.000, 0.000, 1.300, 1.980, 2.57], 
[0.716, 0.567, 0.000, 0.000, 0.000, 0.000, 0.000, 0.00],
[0.000, 0.149, 0.567, 0.000, 0.000, 0.000, 0.000, 0.00],
[0.000, 0.000, 0.149, 0.604, 0.000, 0.000, 0.000, 0.00],
[0.000, 0.000, 0.000, 0.235, 0.560, 0.000, 0.000, 0.00],
[0.000, 0.000, 0.000, 0.000, 0.225, 0.678, 0.000, 0.00],
[0.000, 0.000, 0.000, 0.000, 0.000, 0.249, 0.851, 0.00],
[0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.016, 0.86]])

# TransitionMatrix = np.array([[0.0602, 0.2543,  0], [0.2107, 0, 0.5930], [0.2288, 0, 0.5930]]).T
time = 100

# -------------- Methods to Compute All The Stuff ---------------
def projection(TransitionMatrix, p0_normalized, time):
	generation_list = p0_normalized
	previous_p = p0_normalized 
	population_size = [sum(p0_normalized)]
	for i in range(time):
		next_p = np.matmul(TransitionMatrix, previous_p)
		next_p_normalized = next_p #/sum(next_p)
		population_size = population_size + [sum(next_p)]
		generation_list = np.hstack((generation_list, next_p_normalized))
		previous_p = next_p_normalized

	return generation_list, population_size

def plotGenerations(TransitionMatrix, Total_Projections, Total_Projections_Stable):
	height, _ = TransitionMatrix.shape
	rows = int(np.ceil(height/4))
	fig, ax = plt.subplots(nrows=rows,ncols=4,figsize=(8,8),sharey=True, )
	plt.subplots_adjust(hspace=.5, wspace=0.4)
	for i in range(height):
		ax[int(i/4),i%4].plot(Total_Projections[i,:], label="Random")
		ax[int(i/4),i%4].plot(Total_Projections_Stable[i,:], label="Stable")
		ax[int(i/4),i%4].set_title("Class %i" %(i+1))
		ax[int(i/4),i%4].set_xlabel("Time")
		if (i%4 == 0):
			ax[int(i/4),i%4].set_ylabel("Population Size")
	plt.show()
	return None 

def domEV(TransitionMatrix):
	val, vec = np.linalg.eig(TransitionMatrix)
	index_of_max = np.argmax(val)
	dom_eig, dom_eigvec = val[index_of_max], (np.expand_dims(vec[:,index_of_max], axis=1))/sum(vec[:, index_of_max])
	# Sanity Check that Eig is in fact eig. 
	# print(vec[:,index_of_max], np.matmul(TransitionMatrix,vec[:,index_of_max])/val[index_of_max])
	return dom_eig.real, dom_eigvec.real

def RV(TransitionMatrix):
	_, rv_array = domEV(TransitionMatrix.T)
	rv_array_standardized = rv_array.real/rv_array[0].real
	return rv_array.real, rv_array_standardized.real

def Sensitivity(TransitionMatrix):
	_, reproductive_value_standardized = RV(TransitionMatrix)
	val, vec = domEV(TransitionMatrix)
	numerator = np.matmul(reproductive_value_standardized, vec.T)
	denominator = np.matmul(reproductive_value_standardized.T, vec)
	sens_all = numerator/denominator
	sens_masked = numerator/denominator
	height, width = TransitionMatrix.shape
	for i in range(height):
		for j in range(width):
			if(TransitionMatrix[i,j] == 0):
				sens_masked[i,j] = 0
	return sens_all, sens_masked

def reactivity(TransitionMatrix, initialPop):
	val, _ = domEV(TransitionMatrix)
	reactivity = np.sum(np.matmul(TransitionMatrix/val, initialPop))
	return reactivity

def Inertia(TransitionMatrix, initialPop):
	repval, _ = RV(TransitionMatrix)
	_, vec = domEV(TransitionMatrix)
	numerator = np.matmul(repval.T, initialPop)*sum(vec)
	denominator = np.matmul(repval.T, vec)
	inertia = (numerator/denominator)[0,0].real
	return inertia

def MaxAttenuation(TransitionMatrix, initialPop, time):
	val, _ = domEV(TransitionMatrix)
	_, popSize = projection(TransitionMatrix/val, initialPop, time)
	maxAttenuation = np.min(popSize)
	time_point = np.argmin(popSize)
	return time_point, maxAttenuation

def MaxAmplification(TransitionMatrix, initialPop, time):
	val, _ = domEV(TransitionMatrix)
	_, popSize = projection(TransitionMatrix/val, initialPop, time)
	maxAmplification = np.max(popSize)
	time_point = np.argmax(popSize)
	return time_point, maxAmplification

def StageBiasedProjection(TransitionMatrix, time):
	#Uses Transient Indecies 
	size, _ = TransitionMatrix.shape
	val, vec = domEV(TransitionMatrix)
	for i in range(size):
		initialPopulation = np.zeros(size)
		initialPopulation[i] = 1
		_, popSize = projection(TransitionMatrix/val, initialPopulation, time)

		reac_random = reactivity(TransitionMatrix, initialPopulation)
		inertia_random = Inertia(TransitionMatrix, initialPopulation)
		maxAmp_random_time, maxAmp_random = MaxAmplification(TransitionMatrix, initialPopulation, time)
		maxAtt_random_time, maxAtt_random = MaxAttenuation(TransitionMatrix, initialPopulation, time)
		plt.plot([1],[reac_random], "ro")
		plt.plot([time],[inertia_random], "bo")
		plt.plot([maxAtt_random_time], [maxAtt_random], "go")
		plt.plot([maxAmp_random_time], [maxAmp_random], "yo")
		plt.plot(popSize, label="Population Biased: % i " % i)

	plt.title("Stage Biased Projection")
	plt.xlabel("Time")
	plt.ylabel("Population Size / Growth Rate")
	plt.legend()
	plt.show()
	return None 

def CrazySampling(TransitionMatrix, time, samples):
	size, _ = TransitionMatrix.shape
	val, vec = domEV(TransitionMatrix)
	for i in range(samples):
		initialPopulation = np.random.rand(size,1)
		initialPopulation = initialPopulation/sum(initialPopulation)
		_, popSize = projection(TransitionMatrix/val, initialPopulation, time)	
		plt.plot(popSize,"black", alpha=0.05)
	plt.title("Dynamics of %i Random Initial Populations" %samples)
	plt.xlabel("Time")
	plt.ylabel("Population Size / Growth Rate")
	plt.show()
	return None 

def TransferFunctionForEV(TransitionMatrix):
	growthRatesToZero = np.expand_dims(np.zeros(TransitionMatrix.shape[0]),axis=0)
	noGrowthRateTM = np.vstack((growthRatesToZero, TransitionMatrix[1:,:]))
	fig, ax = plt.subplots(nrows=TransitionMatrix.shape[0],ncols=TransitionMatrix.shape[1] ,figsize=(8,8), sharex=True, sharey=True)
	fig2, ax2 = plt.subplots()
	for i in range(TransitionMatrix.shape[0]):
		for j in range(TransitionMatrix.shape[1]):
			testerTM = TransitionMatrix

			if(noGrowthRateTM[i,j] != 0):
				upper_perturbation_bound = 1 - (sum(noGrowthRateTM[:,j]))
				lower_perturbation_bound = (-1)*noGrowthRateTM[i,j]
				perturb_space = np.linspace(lower_perturbation_bound, upper_perturbation_bound, 20)
				lambs = np.array([])
				for k in range(len(perturb_space)):
					testerTM[i,j] = testerTM[i,j] + perturb_space[k]
					val, vec = domEV(testerTM)
					lambs = np.append(lambs, val)
					testerTM[i,j] = testerTM[i,j] - perturb_space[k]

				ax2.plot(perturb_space, lambs)
				ax[i,j].plot(perturb_space, lambs, label="%i %i" %(i, j))
				ax[i,j].plot()

			if(noGrowthRateTM[i,j] == 0 and TransitionMatrix[i,j] != 0):
				#dealing with growth rates
				if(TransitionMatrix[i,j] > 0.6):
					perturb_space = np.linspace(-0.5,0.5, 20)
				else:
					perturb_space = np.linspace(TransitionMatrix[i,j], 0.5, 20)
				lambs = np.array([])
				for k in range(len(perturb_space)):
					testerTM[i,j] = testerTM[i,j] + perturb_space[k]
					val, vec = domEV(testerTM)
					lambs = np.append(lambs, val)
					testerTM[i,j] = testerTM[i,j] - perturb_space[k]

				ax2.plot(perturb_space, lambs)
				ax2.set_ylabel("Growth Rate/Dominant Eigenvalue")
				ax2.set_xlabel("Perturbation Value")
				ax2.set_title("Non-Linear Perturbation Analysis: Growth Rate")
				ax[i,j].plot(perturb_space, lambs, label="%i %i" %(i, j))
				ax[i,j].plot()

	plt.show(fig)
	plt.show(fig2)

def TransferFunctionForReactivity(TransitionMatrix, initialPopulation):
	growthRatesToZero = np.expand_dims(np.zeros(TransitionMatrix.shape[0]),axis=0)
	noGrowthRateTM = np.vstack((growthRatesToZero, TransitionMatrix[1:,:]))
	fig, ax = plt.subplots(nrows=TransitionMatrix.shape[0],ncols=TransitionMatrix.shape[1] ,figsize=(8,8), sharex=True, sharey=True)
	fig2, ax2 = plt.subplots()
	for i in range(TransitionMatrix.shape[0]):
		for j in range(TransitionMatrix.shape[1]):
			testerTM = TransitionMatrix

			if(noGrowthRateTM[i,j] != 0):
				upper_perturbation_bound = 1 - (sum(noGrowthRateTM[:,j]))
				lower_perturbation_bound = (-1)*noGrowthRateTM[i,j]
				perturb_space = np.linspace(lower_perturbation_bound, upper_perturbation_bound, 20)
				lambs = np.array([])
				for k in range(len(perturb_space)):
					testerTM[i,j] = testerTM[i,j] + perturb_space[k]
					val = reactivity(TransitionMatrix, initialPopulation)
					lambs = np.append(lambs, val)
					testerTM[i,j] = testerTM[i,j] - perturb_space[k]

				ax2.plot(perturb_space, lambs)
				ax[i,j].plot(perturb_space, lambs, label="%i %i" %(i, j))
				ax[i,j].plot()

			if(noGrowthRateTM[i,j] == 0 and TransitionMatrix[i,j] != 0):
				#dealing with growth rates
				if(TransitionMatrix[i,j] > 0.6):
					perturb_space = np.linspace(-0.5,0.5, 20)
				else:
					perturb_space = np.linspace(TransitionMatrix[i,j], 0.5, 20)
				lambs = np.array([])
				for k in range(len(perturb_space)):
					testerTM[i,j] = testerTM[i,j] + perturb_space[k]
					val = reactivity(TransitionMatrix, initialPopulation)
					lambs = np.append(lambs, val)
					testerTM[i,j] = testerTM[i,j] - perturb_space[k]

				ax2.plot(perturb_space, lambs)
				ax2.set_ylabel("Reactivity")
				ax2.set_xlabel("Perturbation Value")
				ax2.set_title("Non-Linear Perturbation Analysis: Reactivity")
				ax[i,j].plot(perturb_space, lambs, label="%i %i" %(i, j))
				ax[i,j].plot()

	plt.show(fig)
	plt.show(fig2)

def Dampening(TransitionMatrix):
	val, vec = np.linalg.eig(TransitionMatrix)
	val.sort()
	print(val)
	dampeningRatio = val.real[-1]/val.real[-2]
	return dampeningRatio


# -------- Use Methods Above for TM & Time -----------
#Eigenvector and eigenvalue information

val, vec = domEV(TransitionMatrix)
print("Growth Rate/Dominant Eigenvalue: ", val, "\nStable Population Structure/Dominant Eigenvector: \n", vec)
vecNew = []
for i in range(len(vec)):
	vecNew.append(vec[i,0])
plt.bar(np.linspace(1,TransitionMatrix.shape[0], TransitionMatrix.shape[0]), vecNew)
plt.title("Stable Poulation Structure Distribution")
plt.ylabel("Population Size")
plt.xlabel("Class")
plt.show()

#Calculate Reproductive Value
reproductive_value, reproductive_value_standardized = RV(TransitionMatrix)
print("RV Array (Standardized): \n", reproductive_value_standardized)


#Sensitivity Analysis
sens_all, sens_masked = Sensitivity(TransitionMatrix)
print("Sensitivity Matrix:\n", sens_all, "\n", sens_masked)

#Elasticity Computation 
elasticity = np.matmul(TransitionMatrix/val, sens_all)
print("Elasticity Matrix: \n" , elasticity)

#Convergence Metrics 
print("Dampening Ratio: ", Dampening(TransitionMatrix))

#Random p0 Generation & Normalization 
length, _ = TransitionMatrix.shape
p0 = np.random.rand(length,1) 
p0_normalized = p0/(sum(p0))

#Compute Random p0 projections
Total_Projections, random_pop_size = projection(TransitionMatrix, p0_normalized, time)

#Compute projectoin of Dominant Eigenvalue
Total_Projections_Stable, ev_pop_size = projection(TransitionMatrix, vec, time)

#Compute projections by disocounting max eigenvalue thereby disentangling asymptotic and transiet dyanmics
Total_Projections_Random_Discounted, discount_random_pop_size = projection(TransitionMatrix/val, p0_normalized, time)
Total_Projections_Stable_Discounted, discount_stable_pop_size = projection(TransitionMatrix/val, vec, time)

#Compute Reactivity 
reac_random = reactivity(TransitionMatrix, p0_normalized)
reac_stable = reactivity(TransitionMatrix, vec)

#Compute Inertia 
inertia_random = Inertia(TransitionMatrix, p0_normalized)
inertia_stable = Inertia(TransitionMatrix, vec.real)

#Compute Max Attenuation
maxAtt_random_time, maxAtt_random = MaxAttenuation(TransitionMatrix, p0_normalized, time)
maxAtt_stable_time, maxAtt_stable = MaxAttenuation(TransitionMatrix, vec, time)

#Compute Max Amplification 
maxAmp_random_time, maxAmp_random = MaxAmplification(TransitionMatrix, p0_normalized, time)
maxAmp_stable_time, maxAmp_stable = MaxAmplification(TransitionMatrix, vec, time)
#Y-Axis is Percentage of population 
plotGenerations(TransitionMatrix, Total_Projections, Total_Projections_Stable)

# Population Size Plot  
plt.plot(random_pop_size, label="Random")
plt.plot(ev_pop_size, label="Stable")
plt.ylabel("Population Size")
plt.xlabel("Time")
plt.title("Population Size v. Time")
plt.legend()
plt.show()

plt.plot(discount_random_pop_size, label="Random")
plt.plot(discount_stable_pop_size, label="Stable")
plt.plot([1,1],[reac_random, reac_stable], "ro", label="Reactivity")
plt.plot([time, time],[inertia_stable, inertia_random], "bo", label="Inertia")
plt.plot([maxAmp_random_time, maxAmp_stable_time], [maxAmp_random, maxAmp_stable], "yo", label="Max Amplification")
plt.plot([maxAtt_random_time, maxAtt_stable_time], [maxAtt_random, maxAtt_stable], "go", label="Max Attenuation")
plt.title("Discounted Population Projections")
plt.ylabel("Population Size / Growth Rate")
plt.xlabel("Time")
plt.legend(loc="right")
plt.show()

#Stage Biased Projections i.e. [0,0,0,0,0,0,0,1] w/ 1 rotating 
StageBiasedProjection(TransitionMatrix, time)

#Sample a bunch of times 
CrazySampling(TransitionMatrix, time, 1000)

#Transfer Functions
TransferFunctionForEV(TransitionMatrix)

TransferFunctionForReactivity(TransitionMatrix, p0_normalized)



