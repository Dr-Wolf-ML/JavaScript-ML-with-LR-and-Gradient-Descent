const tf = require('@tensorflow/tfjs');
const plot = require('node-remote-plot');

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
  learningRate: 0.5,
  iterations: 100,
  batchSize: 50,
  decisionBoundary: 0.5
});

regression.train();

plot({
  name: 'Cost_History_per_iteration',
  title: 'Cost History (Cross Entropy)',
  x: regression.costHistory.reverse().flat(2),
  xLabel: 'Iterations #',
  yLabel: 'Cross Entropy'
})

// Classification Accuracy
console.log(regression.test(testFeatures, testLabels));
