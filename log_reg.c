#define _CRT_SECURE_NO_WARNINGS
#include<stdlib.h>
#include<math.h>
#include<stdio.h>
#include<string.h>
#include<time.h>

#define ROW 4 // num features + 1 for bias term
#define TRAIN_SIZE 20000
#define TEST_SIZE 4000
#define VERBOSE 0
#define L2 0.1
#define TOL 0.0001


float sig(float *w, float *x, int m) {
	float s = 0;
	for (int i = 0; i < m; i++) {
		s -= w[i] * x[i];
	}
	return 1 / (1 + expf(s));
}

/*
float log_likelihood(float **X, float *Y, float *W, int m, int n) {
float ll = 0;
for (int i = 0; i < n; i++) {
float dot = 0;
for (int j = 0; j < m; j++) {
dot += W[j] * X[i][j];
}
ll += Y[i] * dot - logf(1 + expf(dot));
}
return ll;
}
*/

float* error(float *X, float *Y, float *W, float *err, int m, int n) {
	float *temp = calloc(m, sizeof(float));
	for (int i = 0; i < n; i++) {
		//memcpy(temp,X[m*i],m*sizeof(float));
		for (int j = 0; j < m; j++) {
			temp[j] = X[m*i + j];
		}
		err[i] = Y[i] - sig(W, temp, m);
	}
	free(temp);
	return err;
}

float MSE(float *X, float *Y, float *W, float *err, int m, int n) {
	float *temp = calloc(m, sizeof(float));
	for (int i = 0; i < n; i++) {
		//memcpy(temp,X[m*i],m*sizeof(float));
		for (int j = 0; j < m; j++) {
			temp[j] = X[m*i + j];
		}
		err[i] = Y[i] - sig(W, temp, m);
	}
	free(temp);
	float mse = 0;
	for (int i = 0; i < n; i++){
		mse += err[i] * err[i];
	}
	return mse/n;
}

float* gradient(float *X, float *Y, float *W, float *grad, float *err, int m, int n) {
	err = error(X, Y, W, err, m, n);

	float val = 0;
	/*	*/

	for (int i = 0; i < n; i++) {
		val += X[i*m] * err[i];
	}
	grad[0] = 0;//val; // no L2 regularizaion for bias term

	for (int j = 1; j < m; j++) {
		val = 0;
		for (int i = 0; i < n; i++) {
			val += X[i*m + j] * err[i];
		}
		grad[j] = val - L2*W[j]; // added L2 regularizaion
	}
	return grad;
}


float* train(int m, int n, float *X, float *Y, float *W, float lr, int n_epochs) {
	// define all functions within...
	// X is an array of n rows and m columns
	float *grad = calloc(m, sizeof(float)); // all additional memory created here
	float *err = calloc(n, sizeof(float));
	float temp = 1000.0;
	float mse = 1000.0;
	int steps = 0;
	//split data
	int train_size = 0.9 * n;
	float *X_train = calloc(train_size*m, sizeof(float));
	float *Y_train = calloc(train_size, sizeof(float));
	float *X_val = calloc((n-train_size)*m, sizeof(float));
	float *Y_val = calloc((n-train_size), sizeof(float));

	for (int i=0; i<train_size; i++){ //copy train data
		X_train[i] = X[i];
		Y_train[i] = Y[i];
	}
	for (int i=0; i<n-train_size; i++){ //copy test data
		X_val[i] = X[i+train_size];
		Y_val[i] = Y[i+train_size];
	}

	W[0] = 1; // set bias fixed to 1
	while (mse > TOL && steps < n_epochs) {
		steps ++;
		grad = gradient(X_train, Y_train, W, grad, err, m, train_size);
		for (int j = 0; j < m; j++) {
			W[j] += lr * grad[j]; // we ascend the gradient here.
		}
		temp = MSE(X_val, Y_val, W, err, m, n-train_size);
		if (temp > mse){ // early stopping
			// printf("\nEarly Stopping at epoch no. %d...\n",steps);
			steps = n_epochs;
		}
		else{
			mse = temp;
		}
	}

	// free all memory
	free(grad);
	free(err);
	free(X_train);
	free(Y_train);
	free(X_val);
	free(Y_val);
	return W;
}


float* evaluate(int m, int n, float *X, float *Y, float *W) {
	// X is an array of n rows and m columns
	//W weights
	float *temp = calloc(m, sizeof(float));
	float prob;
	int tp = 0; // true positives
	int fp = 0; // false positives
	int fn = 0; // false negatives
	for (int i = 0; i<n; i++) {
		for (int j = 0; j < m; j++) {
			temp[j] = X[m*i + j];
		}
		prob = sig(W, temp, m);
		if (prob > 0.5) { // it is in positive class
			if (Y[i] == 1) {
				tp++;
			}
			else {
				fp++;
			}
		}
		else {
			if (Y[i] == 1) {
				fn++;
			}
		}
	}

	float precision = 100.0 * tp / (tp + fp);
	float recall = 100.0 * tp / (tp + fn);
	float f1score = 2 * precision * recall / (precision + recall);
	printf("\n\nPrecision: %.2f%% \nRecall: %.2f%% \nF1-score: %.2f%% \n\n", precision, recall, f1score);

	// free all memory
	free(temp);
	return W;
}

const char* getfield(char* line, int num) {
	const char* tok;
	for (tok = strtok(line, ",");
		tok && *tok;
		tok = strtok(NULL, ",\n"))
	{
		if (!--num)
			return tok;
	}
	return NULL;
}

void load_data(float *X_train, float *Y_train, float *X_test, float *Y_test) {

	// FILE *csvFile = fopen("creditcard.csv");
	FILE* csvfile = fopen("skin.csv", "r");

	if (!csvfile){
		exit(0);
	}
	char line[1024];
	int count = 0;
	while (count < TRAIN_SIZE && fgets(line, 1024, csvfile)) {
		// printf("Field 3 would be %s\n", getfield(tmp, 3));
		// NOTE strtok clobbers tmp
		char* tmp;
		X_train[count * ROW] = 1;
		for (int i = 1; i <= ROW-1; i++) {
			tmp = _strdup(line);
			X_train[count * ROW + i] = atof(getfield(tmp, i));
		}
		tmp = _strdup(line);
		Y_train[count] = atof(getfield(tmp, ROW));
		free(tmp);

		count++;
		if (VERBOSE && count % 1000 == 0) {
			printf("\n%d records taken", count);
		}
	}

	count = 0;
	while (count < TEST_SIZE && fgets(line, 1024, csvfile)) {
		// printf("Field 3 would be %s\n", getfield(tmp, 3));
		// NOTE strtok clobbers tmp
		char* tmp;
		X_test[count * ROW] = 1;
		for (int i = 1; i <= ROW-1; i++) {
			tmp = _strdup(line);
			X_test[count * ROW + i] = atof(getfield(tmp,i));
		}
		tmp = _strdup(line);
		Y_test[count] = atof(getfield(tmp,ROW));
		free(tmp);
		count++;
		if (VERBOSE && count % 1000 == 0) {
			printf("\n%d records taken", count + TRAIN_SIZE);
		}
	}
	fclose(csvfile);

}

void save_to_csv(float *arr, int n) {
	FILE* csvfile = fopen("weights.txt", "w+");
	for (int i = 0; i < n; i++) {
		fprintf(csvfile, "%f ", arr[i]);
	}


	fclose(csvfile);
}

int main() {// to test it out
	float lr = 0.001;
	int n_epochs = 10000;
	float *X_train, *Y_train, *X_test, *Y_test;

	X_train = calloc(ROW * TRAIN_SIZE, sizeof(float));
	Y_train = calloc(TRAIN_SIZE, sizeof(float));
	X_test = calloc(ROW * TEST_SIZE, sizeof(float));
	Y_test = calloc(TEST_SIZE, sizeof(float));

	load_data(X_train, Y_train, X_test, Y_test);
	float *W = calloc(ROW, sizeof(float));

	clock_t begin = clock();
	train(ROW, TRAIN_SIZE, X_train, Y_train, W, lr, n_epochs);
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("\n\nTotal time spent training:%.2f seconds", time_spent);
	printf("\n\nSome Weights: \n");
	for (int i = 0; i < ROW; i++) {
		printf("%.2f\n", W[i]);
	}
	evaluate(ROW, TEST_SIZE, X_test, Y_test, W);
	save_to_csv(W, ROW); //save the weights to a csv file

	free(W);
	free(X_train);
	free(Y_train);
	free(X_test);
	free(Y_test);
}

/*
int main() {// to test it out
int m = 2, n = 2;
float X[4] = {  0,1 , 1,0  };
float Y[2] = { 0,1 };
float W[2] = { 0.1,0.2 };
float lr = 0.001;
train(m, n, X, Y, W, lr, 10000);
printf("\nWeights: %f \t %f \n\n", W[0], W[1]);
return 1;

}
*/