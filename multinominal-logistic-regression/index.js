// require('@tensorflow/tfjs-node');  // threw error in OSX.14.beta.6
const tf = require('@tensorflow/tfjs');
const plot = require('node-remote-plot');

const loadCSV = require('../load-csv');
const LogisticRegression = require('./logistic-regression');

const {features, labels, testFeatures, testLabels} = loadCSV('../data/cars.csv', {
  dataColumns: ['horsepower', 'displacement', 'weight'],
  labelColumns: ['mpg'],
  converters: {
    mpg: (value) => {
      const mpg = parseFloat(value);

      return mpg <= 15 ? [1, 0, 0] : mpg <= 30 ? [0, 1, 0] : [0, 0, 0];
    }
  },
  shuffle: true,
  splitTest: 50
});

const regression = new LogisticRegression(features, labels.flat(1), {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 10,
  calculationMode: 'conditional'  // 'marginal' or 'conditional' === sigmoid vs softmax
});

regression.train();

// Classification Accuracy
console.log(regression.test(testFeatures, testLabels.flat(1)));


// plot({
//   name: 'Cost_History_per_iteration',
//   title: 'Cost History (Cross Entropy)',
//   x: regression.costHistory.reverse().flat(2),
//   xLabel: 'Iterations #',
//   yLabel: 'Cross Entropy'
// })
