const tf = require('@tensorflow/tfjs');

const LinearRegression = require('./linear-regression');

const loadCSV = require('./load-csv');

let {features, labels, testFeatures, testLabels} = loadCSV('./cars.csv', {
  dataColumns: ['horsepower'],
  labelColumns: ['mpg'],
  converters: {},
  shuffle: true,
  splitTest: 50
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.0001,
  iterations: 100
});

regression.train();

console.log('Updated m: ', regression.m, '  Updated b: ', regression.b);
