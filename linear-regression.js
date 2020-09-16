const tf = require('@tensorflow/tfjs');

class LinearRegression {
  constructor(features, labels, options) {
    // expected to be passed as tensorflow Tensors:
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.mseHistory = [];
    this.bHistory = [];

    this.options = Object.assign({
      learningRate: 0.1,
      iterations: 1000
    }, options)

    this.weights = tf.zeros([this.features.shape[1],1]);
  }

  gradientDescent() {
    const currentGuesses = this.features.matMul(this.weights);
    const differences = currentGuesses.sub(this.labels);
    const slopes = this.features  // slopes is sometimes called 'gradients'
      .transpose()
      .matMul(differences)
      .div(this.features.shape[0])
      // .mul(2) is require in the original equation, but doesn't add any real value here !!

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.bHistory.push(this.weights.arraySync()[0][0]);
      this.gradientDescent();
      this.recordMSE();
      this.updateLearningRate();
    }
  }

  test(testFeatures, testLabels) {
    testFeatures = this.processFeatures(testFeatures);
    testLabels = tf.tensor(testLabels);

    const predictions = testFeatures.matMul(this.weights);

    // Sum of Squares Residual
    const SSres = testLabels.sub(predictions)
      .pow(2)
      .sum()
      .arraySync()

    // Sum of Squares Total
    const SStot = testLabels
      .sub(testLabels.mean())
      .pow(2)
      .sum()
      .arraySync()

    // R^2 (= Coefficient of Determination)
    return 1 - SSres / SStot;
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

  recordMSE() {
    const mse = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .arraySync()

    this.mseHistory.unshift(mse);
  }

  updateLearningRate() {
    if (this.mseHistory.length < 2) {
      return;
    } 
    
    if (this.mseHistory[0] > this.mseHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LinearRegression;
