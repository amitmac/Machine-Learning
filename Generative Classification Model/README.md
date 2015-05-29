# Generative Classification

Learn a model with class-conditional densities and class priors
and use the parameters of such a model to do the prediction.

Likelihood / Class-conditional density

	P(x|c=0) = N(mu0,sigma0),
	
	P(x|c=1) = N(mu1,sigma1)

Priors

	P(c=0) = N0 / (N0 + N1),
	
	P(c=1) = N1 / (N0 + N1)

Given a training set, we calculate the estimates of the model parameters and then use the generative or discriminant functions to predict the class for testing data.

We calculate p(c=0|x) and p(c=1|X) which are the discriminant function. If p(c=0|x) > p(c=1|X) then x belongs to class c=0 else x belongs to class c=1.

We take the discriminant function as the log posterior probability

	log(P(c|x)) = log(P(x|c)) + log(P(c)) + constant
	
	For a gaussian pdf,
	
		log(P(c|x)) = 
			(-1/2)(x - mu)' * inverse(sigma) * (x - mu) - (1/2) + log(inverse(sigma)) + log(P(c))

After calculating parameters, just put them into the above equation for each class and compare to predict the class.
