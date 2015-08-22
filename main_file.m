% Machine Learning Multinomial Logistic Regression
% Initialization
clear; close all; clc
% There are 4 classes into which data has to be classified.

for run=1:4

%% Load Data
%  The first two columns contains the pulse rate and age of the
%  person, the third column
%  contains the label.

%  Saving the training set filenames in 'name' variable 
%  and then loading them	
 
 	name = 'class';
	name = strcat(name,48+run);
	name = strcat(name,'.txt');
	data = load(name);								
	X = data(:, [1, 2]); y = data(:, 3);
							

%% ==================== Part 1: Plotting ====================
%  We start the exercise by first plotting the data to understand the 
%  the problem we are working with.

fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);

plotData(X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Pulse rate')
ylabel('Age')

% Specified in plot order
legend('Belongs to the class', 'Does not belong to the class')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ============ Part 2: Compute Cost and Gradient ============
%  In this part of the exercise, you will implement the cost and gradient
%  for logistic regression.

%  Mapping the feature set to a higher dimension or
%  in other words converting the features to a quadratic form

	q1 = data(:,1);
	q2 = data(:,2);
	X = mapFeature(q1,q2);	

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ============= Part 3: Optimizing using fminunc  =============
%  In this exercise, you will use a built-in function (fminunc) to find the
%  optimal parameters theta.

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);


%We are still working on the plotting of decision boundary for   %each class, as it works fine for the first and the last class   %which has only one linear decision boundary but the middle      %classes have minimum two decision boundaries each!


% Plot Boundary
%plotDecisionBoundary(theta, X, y);

% Put some labels 
%hold on;
% Labels and Legend
%xlabel('Pulse Rate')
%ylabel('Age')

% Specified in plot order
%legend('Belongs to the class', 'Does not belong to the class')
%hold off;

%fprintf('\nProgram paused. Press enter to continue.\n');
%pause;

%% ============== Part 4: Predict and Accuracies ==============
%  After learning the parameters, you'll like to use it to predict the outcomes
%  on unseen data. In this part, we will use the logistic regression model
%  to predict the probability that a person with some pulse rate
%  and some age belongs to which specific class.
%
%  Furthermore, we compute the training and test set accuracies of 
%  our model.
%

% Compute accuracy on our training set
p = predict(theta, X);

%  For each classifier, its theta values are stored in
%  'history_theta' as a particular row. 
	
	if (run==1)

		history_theta=theta';
	else
		history_theta=[history_theta; theta'];
	end


fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

end

%  This is to verify the classifier values i.e. to which class
%  the test set belongs to and the probability of it belonging to 
%  that specific class.

%  Feature readings have to given in the following format
%  '[111 21]'
%  where 111 corresponds to the pulse rate and 21 corresponds
%  to the age.

for iteration = 1:20
	w1 = input("\n\n\nEnter your pulse rate reading : ");
	w2 = input("Enter your age : ");
	input_data = mapFeature(w1,w2);	
	pc = predictOneVsAll(history_theta,input_data);

	if (pc==1)
		fprintf('You belong to class 1 with a prbability equal to pval and if you do not exercise very regulary, then please do visit a doctor immediately because you have a abnormal pulse rate reading!\n');

	else if (pc==2)
		fprintf('You belong to class 2 with a prbability equal to pval and you have a normal reading. You are expected to be in a resting/calm state.\n');

	else if (pc==3)
		fprintf('You belong to class 3 with a prbability equal to pval and you have a reading of an excited person!\n');

	else if (pc==4)
		fprintf('You belong to class 4 with a prbability equal to pval and you have a reading of a person under exertion or work out!\n');
	end
end
end
end
end
