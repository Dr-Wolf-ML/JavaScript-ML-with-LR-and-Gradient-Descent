const tf = require('@tensorflow/tfjs');
const plot = require('node-remote-plot');

const LinearRegression = require('./linear-regression');
const loadCSV = require('../load-csv');

let {features, labels, testFeatures, testLabels} = loadCSV('../data/cars.csv', {
  dataColumns: ['horsepower', 'weight', 'displacement'],
  labelColumns: ['mpg'],
  converters: {},
  shuffle: true,
  splitTest: 50
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 5,
  batchSize: 30       // set to 1 for Stochastic Gradient Descent
});

regression.train();

// To prevent Plot throwing an error, ensure x === y
console.log('MSE History length: ', regression.mseHistory.length);
console.log('B History length: ', regression.bHistory.length);

const plotA = () => {
    try {
      plot({
        name: 'MSE_per_Iteration',
        title: 'MSE per Iteration',
        x: regression.mseHistory,
        xLabel: 'Iterations #',
        yLabel: 'MSE - Mean Squared Error'
      });
      console.log('Plot: MSE per Iteration - Succeeded');
    } catch(err) {
      console.log('Plot MSE per Iteration failed with Error Message: ', err.message);
    }
};

const plotB = async () => {
    try {
      plot({
        name: 'MSE_per_B',
        title: 'MSE per B',
        x: regression.bHistory.slice(0, regression.mseHistory.length),
        y: regression.mseHistory.reverse(),
        xLabel: 'value of B',
        yLabel: 'MSE - Mean Squared Error'
      });
      console.log('Plot: MSE per B - Succeeded');
    } catch(err) {
      console.log('Plot MSE per B failed with Error Message: ', err.message);
    }
};

plotA();
plotB();

// Accuracy R2
const r2 = regression.test(testFeatures, testLabels);
console.log('R2: ', r2);

// Use & Predict
// regression.predict([
//   [120, 2, 380],
//   [135, 2.2, 420]
// ]).print();
