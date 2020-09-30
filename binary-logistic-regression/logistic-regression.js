const tf = require('@tensorflow/tfjs');

class LogisticRegression {
  constructor(features, labels, options) {
    // expected to be passed as tensorflow Tensors:
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.costHistory = [];  // 'cost...' is the scientific term used in literature for Cross Entropy
    this.bHistory = [];

    this.options = Object.assign({
      learningRate: 0.1,
      iterations: 100,
      batchSize: 50,
      decisionBoundary: 0.5
    }, options)

    this.weights = tf.zeros([this.features.shape[1],1]);
  }

  gradientDescent(features, labels) {
    const currentGuesses = features
      .matMul(this.weights)
      .sigmoid();
    const differences = currentGuesses.sub(labels);
    const slopes = features  // slopes is sometimes called 'gradients'
      .transpose()
      .matMul(differences)
      .div(features.shape[0])
      // .mul(2) is required in the original equation, but doesn't add any real value here !!

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
      );

    const {batchSize} = this.options;

    const trainingSlice = (tensorSet, index) => {
      return tensorSet.slice(
        [index * batchSize, 0],
        [batchSize, -1]
      );
    };

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const featureSlice = trainingSlice(this.features, j);
        const labelSlice = trainingSlice(this.labels, j);

        this.bHistory.push(this.weights.arraySync()[0][0]);
        this.gradientDescent(featureSlice, labelSlice);
      }
      
      this.recordCost();
      this.updateLearningRate();
    }
  }

  predict(observations) {
    return this.processFeatures(observations)
      .matMul(this.weights)
      .sigmoid()
      .greater(this.options.decisionBoundary)
      .cast('float32');
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures);
    testLabels = tf.tensor(testLabels);

    const incorrect = predictions
      .sub(testLabels)
      .abs()
      .sum()
      .arraySync();

    return (predictions.shape[0] - incorrect) / predictions.shape[0];
  }

  processFeatures(features) {
    features  = tf.tensor(features);

    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.standardize(features);
    }

    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  standardize(features) {
    const {mean, variance} = tf.moments(features, 0);

    this.mean = mean;
    this.variance = variance;

    return features.sub(mean).div(variance.pow(0.5));
  }

  recordCost() {
    const guesses = this.features
      .matMul(this.weights)
      .sigmoid();

    const termOne = this.labels
      .transpose()
      .matMul(
        guesses
          .log()
        );

    const termTwo = this.labels
      .mul(-1)
      .add(1)
      .transpose()
      .matMul(
        guesses
          .mul(-1)
          .add(1)
          .log()
        );

    const cost = termOne
      .add(termTwo)
      .div(this.features.shape[0])
      .mul(-1)
      .arraySync();
   
    this.costHistory.unshift(cost);
  }

  updateLearningRate() {
    if (this.costHistory.length < 2) {
      return;
    } 
    
    if (this.costHistory[0] > this.costHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LogisticRegression;
