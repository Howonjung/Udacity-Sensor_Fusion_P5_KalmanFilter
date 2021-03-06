#ifndef UKF_H
#define UKF_H

#include "Eigen/Dense"
#include "measurement_package.h"
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
 public:
  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package, std::ofstream& radarDataFile);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  // Generate augmented sigma points.
  void AugmentedSigmaPoints(Eigen::MatrixXd* Xsig_out);
  // Predict sigma points which is the process getting sigma points through non-linear prediction function
  void SigmaPointPrediction(Eigen::MatrixXd &Xsig_aug,Eigen::MatrixXd* Xsig_pred_, double delta_t);
  // Calculate mean and state covarian in prediction step using predicted sigma point
  void PredictMeanAndCovariance(Eigen::VectorXd* x_pred,  Eigen::MatrixXd* P_pred);
  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package, std::ofstream& sensorDataFile);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package, std::ofstream& sensorDataFile);
  // Projects prediction result into measuremnt space
  void PredictRadarMeasurement(VectorXd* z_pred_, MatrixXd* S_pred_, MatrixXd *Zsig_);
  void PredictLidarMeasurement(VectorXd* z_pred_, MatrixXd* S_pred_, MatrixXd *Zsig_);
  // Update state and state covariance matrix using Kalman Gain
  void UpdateState(MeasurementPackage meas_package, MatrixXd Zsig_pred, VectorXd z_pred, MatrixXd S_pred, VectorXd* x_pred, MatrixXd* P_pred, std::ofstream& sensorDataFile);

  // initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  // if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  // if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // predicted sigma points matrix
  Eigen::MatrixXd Xsig_pred_;

  // time when the state is true, in us
  long long time_us_;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  // Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  // Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  // Radar measurement noise standard deviation radius in m
  double std_radr_;

  // Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  // Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  // Weights of sigma points
  Eigen::VectorXd weights_;
  bool bWeightInit;

  // State dimension
  int n_x_;

  // Augmented state dimension
  int n_aug_;

  // Sigma point spreading parameter
  double lambda_;
};

#endif  // UKF_H