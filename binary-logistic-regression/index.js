const tf = require('@tensorflow/tfjs');
const plot = require('node-remote-plot');
const _ = require('lodash');

const loadCSV = require('../load-csv');
const LogisticRegression = require('./logistic-regression');

const {features, labels, testFeatures, testLabels} = loadCSV('../data/cars.csv', {
  dataColumns: ['horsepower', 'displacement', 'weight'],
  labelColumns: ['passedemissions'],
  converters: {
    passedemissions: (value) => {
      return value === 'TRUE' ? 1 : 0;
    }
  },
  shuffle: true,
  splitTest: 50
});

const regression = new LogisticRegression(features, labels, {
  learningRate: 1,
  iterations: 100,
  batchSize: 20,
  decisionBoundary: 0.5
});

regression.train();

console.log('costHistory: ', regression.costHistory.map(image => _.flatMap(image)).reverse());

plot({
  name: 'Cost_History_per_iteration',
  title: 'Cost History (Cross Entropy)',
  x: regression.costHistory.map(image => _.flatMap(image)).reverse(),
  xLabel: 'Iterations #',
  yLabel: 'Cross Entropy'
})

// Classification Accuracy
console.log('Classification Accuracy: ', regression.test(testFeatures, testLabels));
