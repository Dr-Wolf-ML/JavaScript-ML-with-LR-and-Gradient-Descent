const tf = require('@tensorflow/tfjs');
const plot = require('node-remote-plot');

const LinearRegression = require('./linear-regression');
const loadCSV = require('./load-csv');

let {features, labels, testFeatures, testLabels} = loadCSV('./cars.csv', {
  dataColumns: ['horsepower', 'weight', 'displacement'],
  labelColumns: ['mpg'],
  converters: {},
  shuffle: true,
  splitTest: 50
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 3,
  batchSize: 10       // set to 1 for Stochastic Gradient Descent
});

regression.train();

// Select either one plot at a time

plot({
  x: regression.mseHistory.reverse(),
  xLabel: 'Iterations #',
  yLabel: 'MSE - Mean Squared Error'
})

// plot({
//   x: regression.bHistory,
//   y: regression.mseHistory.reverse(),
//   xLabel: 'value of B',
//   yLabel: 'MSE - Mean Squared Error'
// });

// Accuracy R2
const r2 = regression.test(testFeatures, testLabels);
console.log('R2: ', r2);

// Use & Predict
regression.predict([
  [120, 2, 380],
  [135, 2.2, 420]
]).print();
