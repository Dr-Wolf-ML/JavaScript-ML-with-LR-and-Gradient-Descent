// require('@tensorflow/tfjs-node');  // threw error in OSX.14.beta.6
const tf = require('@tensorflow/tfjs');
const plot = require('node-remote-plot');
const _ = require('lodash');

const mnist = require('mnist-data');

const LogisticRegression = require('./logistic-regression');

function loadData() {
  const mnistData = mnist.training(0,60000);

  const features = mnistData.images.values.map(image => _.flatMap(image));
  const encodedLabels = mnistData.labels.values.map(value => {
    const imageRow = new Array(10).fill(0);
    imageRow[value] = 1;
    return imageRow;
  });

  return {features, labels: encodedLabels};
};

const {features, labels} = loadData();

// console.log('features: ', features);
// console.log('mnistData.labels.value: ',mnistData.labels.values);
// console.log('encodedLabels: ', encodedLabels);

const regression = new LogisticRegression(features, labels, {
  learningRate: 0.1,
  iterations: 200,
  batchSize: 100,
  classificationFunction: 'softmax'  // select 'softmax' or 'sigmoid'
});

regression.train();

const testMnistData = mnist.testing(0,2000);
const testFeatures = testMnistData.images.values.map(image => _.flatMap(image));
const testEncodedLabels = testMnistData.labels.values.map(value => {
  const imageRow = new Array(10).fill(0);
  imageRow[value] = 1;
  return imageRow;
});

const accuracy = regression.test(testFeatures, testEncodedLabels);
console.log('accuracy: ', accuracy);

console.log('regression.costHistory: ', regression.costHistory);

const plotTitle = regression.options.classificationFunction === 'softmax' ? 'Softmax' : 'Sigmoid';

plot({
  name: 'Cost_History_per_iteration',
  title: `Cost History (Cross Entropy - ${plotTitle})`,
  x: regression.costHistory.map(cost => _.flatMap(cost)).reverse(),
  xLabel: 'Iterations #',
  yLabel: `Cross Entropy - ${plotTitle}`
})
