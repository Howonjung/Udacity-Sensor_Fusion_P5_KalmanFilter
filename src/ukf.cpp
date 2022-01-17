#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using std::cout;
using std::endl;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);
  x_.fill(0.0);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  // Initialize state covariance matrix as identity matrix
  P_ << 1,0,0,0,0,
        0,1,0,0,0,
        0,0,1,0,0,
        0,0,0,1,0,
        0,0,0,0,1;
          
  // TERMS FOR NEEDING A TUNING!
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2.0;
  // // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 3.0;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
   
   // set augmented dimension  
   n_aug_ = 7;
   // set lambda 
   lambda_ = 3 - n_aug_;
   // State dimension
   n_x_ = 5;
   // Check whether weights is initialized or not
   bWeightInit = false;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package, std::ofstream& sensorDataFile) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if (!is_initialized_) {
    cout << "UKF Filter Initialization " << endl;

    // set the state for initial step based on measurement type (MeasurementPackage::RADAR, MeasurementPackage::LASER)
    if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
      // x_ : [pos1 pos2 vel_abs yaw_angle yaw_rate]
      x_ << meas_package.raw_measurements_[0]*cos(meas_package.raw_measurements_[1]), 
            meas_package.raw_measurements_[0]*sin(meas_package.raw_measurements_[1]), 
            0, 
            0,
            0;    
    }
    else{ // MeasurementPackage::LASER 
      // x_ : [pos1 pos2 vel_abs yaw_angle yaw_rate]
      x_ << meas_package.raw_measurements_[0], 
            meas_package.raw_measurements_[1], 
            0, 
            0,
            0;
    }
	  cout << "x_: \n"<< x_ << endl;
    cout << "P_: \n"<< P_ << endl;
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;

    return;
  }

  // compute the time elapsed between the current and previous measurements
  // dt - expressed in seconds
  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;
  
  // prediction step
  Prediction(dt);

  // measurement step after initializing weights for sigma point
  if(bWeightInit)
    if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) UpdateRadar(meas_package, sensorDataFile);
    if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) UpdateLidar(meas_package, sensorDataFile);
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  // process prediction when delta_t is non-zero
  if (delta_t ==0){}
  else{
    	// std::cout << "============ Prediction starts ============" <<std::endl;
	    // cout << "x_pred: \n"<< x_ << endl;
      // cout << "P_pred: \n"<< P_ << endl;
      // Initialize Xsig_aug(x augmented sigma points) & Xsig_pred_(prediction result of augmented sigma point)
      MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
      Xsig_aug.fill(0.0);
      Xsig_pred_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
      Xsig_pred_.fill(0.0);

      // generate augmented sigma points
      AugmentedSigmaPoints(&Xsig_aug);
      // pass sigma points through non-linear prediction model for sigma point prediction
      SigmaPointPrediction(Xsig_aug, &Xsig_pred_, delta_t);
      // calculate mean and state covariance in prediction step using predicted sigma points
      PredictMeanAndCovariance(&x_, &P_);
      
      // std::cout << "Updated state x: " << std::endl << x_ << std::endl;
      // std::cout << "Updated state covariance P: " << std::endl << P_ << std::endl;

	    // std::cout << "============ Prediction ends ============" <<std::endl;
  }
}

void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {

  // create augmented mean vector (add two more noise terms)
  VectorXd x_aug = VectorXd(n_x_ + 2);
  x_aug.fill(0.0);
  // create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0; // process noise vector, Zero Mean Gaussian Noise !!!
  x_aug(6) = 0; // process noise vector, Zero Mean Gaussian Noise !!!

  // create augmented state covariance (add two more noise terms for row and colmuns)
  MatrixXd P_aug = MatrixXd(n_x_ + 2, n_x_ + 2);
  // create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_; // process noise covariance term
  P_aug(6,6) = std_yawdd_*std_yawdd_; // process noise covariance term

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.fill(0.0);

  // create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; ++i) {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

  // print result
  //   std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;

  // write result
  *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(Eigen::MatrixXd &Xsig_aug, Eigen::MatrixXd* Xsig_pred_, double delta_t){

  // create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x_ , 2*n_aug_+ 1);
  Xsig_pred.fill(0.0);

  // squared term of delta_t
  double delta_tSq = pow(delta_t,2);

  // predict sigma points
  VectorXd u = VectorXd(n_x_); // control term in prediction model
  VectorXd noise = VectorXd(n_x_); // noise term in prediction model
  double p_x,p_y,v,yaw,yawVel,nu_a,nu_yawAcc;
  for (int i=0; i<2 * n_aug_ + 1; i++){
    p_x = Xsig_aug(0,i);
    p_y = Xsig_aug(1,i);
    v = Xsig_aug(2,i);
    yaw = Xsig_aug(3,i);
    yawVel = Xsig_aug(4,i);
    nu_a = Xsig_aug(5,i); 
    nu_yawAcc = Xsig_aug(6,i);
    
    noise << 0.5*delta_tSq*cos(yaw)*nu_a,
             0.5*delta_tSq*sin(yaw)*nu_a,
             delta_t*nu_a,
             0.5*delta_tSq*nu_yawAcc,
             delta_t*nu_yawAcc;
    // avoid division by zero
    if(yawVel == 0 ){ // check whether yaw rate is zero or not 
        u << v*cos(yaw)*delta_t,
             v*sin(yaw)*delta_t,
             0.0,
             0.0,
             0.0;
    }
    else{
        u << v/yawVel*(sin(yaw+yawVel*delta_t) - sin(yaw)),
             v/yawVel*(-cos(yaw+yawVel*delta_t) + cos(yaw)),
             0.0,
             yawVel*delta_t,
             0.0;
    }
    // write predicted sigma points into right column
    Xsig_pred.col(i) = Xsig_aug.col(i).head(5) + u + noise;
    // cout <<"Xsig_aug.col(i).head(5): \n" << Xsig_aug.col(i).head(5) << endl;
  }
  
  // print result
  // std::cout << "=======================sigma prediction result=======================" << endl;
  // std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;
  // std::cout << "=======================sigma prediction result ends =======================" << endl <<endl;

  // write result
  *Xsig_pred_ = Xsig_pred;
}

void UKF::PredictMeanAndCovariance(Eigen::VectorXd* x_pred,  Eigen::MatrixXd* P_pred){

  // If weights has't been initialize, then update weights.
  if(!bWeightInit){
    weights_ = VectorXd(2*n_aug_+1);
      // set weights
    for (int i=0; i<2*n_aug_+1; i++){
      if (i==0)
        weights_(i) = lambda_ / (lambda_+n_aug_);
      else 
        weights_(i) = 1.0 / (2.0*(lambda_+n_aug_));
    }
    bWeightInit = true;
  }
  // create vector for predicted state
  VectorXd x = VectorXd(n_x_);
  x.fill(0.0);

  // create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);
  P.fill(0.0);

  // cout << "weight \n" << weights << endl;
  // predict state mean
  for (int j=0; j<2*n_aug_+1; j++){
    x += weights_(j)*Xsig_pred_.col(j);
  }
  // cout <<"x: \n" << x << endl;
  // predict state covariance matrix
  for (int k=0; k<2*n_aug_+1; k++){
    // state difference
    VectorXd x_diff = Xsig_pred_.col(k) - x;
    // Make sure you always normalize when you calculate the difference between angles.
    // Angel normalization!
    while(x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while(x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    P += weights_(k)*x_diff*x_diff.transpose();
  }

  // print result
  // std::cout << "Predicted state" << std::endl;
  // std::cout << x << std::endl;
  // std::cout << "Predicted covariance matrix" << std::endl;
  // std::cout << P << std::endl;

  // write result
  *x_pred = x;
  *P_pred = P;
}

void UKF::UpdateLidar(MeasurementPackage meas_package, std::ofstream& sensorDataFile) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  // projects prediction results into measurement space
  VectorXd z_pred = VectorXd(2);
  z_pred.fill(0.0);
  MatrixXd S_pred = MatrixXd(2, 2);
  S_pred.fill(0.0);
  MatrixXd Zsig_ = MatrixXd(2, 2 * n_aug_ + 1);
  Zsig_.fill(0.0);

  // PredictLidarMeasurement(&z_pred, &S_pred, &Zsig_);
  // update state and state covariance matrix using Kalman Gain
  UpdateState(meas_package, Zsig_, z_pred, S_pred, &x_, &P_, sensorDataFile);
}

void UKF::UpdateRadar(MeasurementPackage meas_package, std::ofstream& sensorDataFile) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  // projects prediction results into measurement space
  VectorXd z_pred = VectorXd(3);
  z_pred.fill(0.0);
  MatrixXd S_pred = MatrixXd(3, 3);
  S_pred.fill(0.0);
  MatrixXd Zsig_ = MatrixXd(3, 2 * n_aug_ + 1);
  Zsig_.fill(0.0);

  PredictRadarMeasurement(&z_pred, &S_pred, &Zsig_);
  // update state and state covariance matrix using Kalman Gain
  UpdateState(meas_package, Zsig_, z_pred, S_pred, &x_, &P_, sensorDataFile);
  
}
void UKF::PredictLidarMeasurement(VectorXd* z_pred_, MatrixXd* S_pred_, MatrixXd *Zsig_){
  
  // set measurement dimension, lidar can measure px,py
  int n_z = 2;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0.0);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);

  // noise covariance matrix S
  MatrixXd R = MatrixXd(n_z,n_z);
  R << pow(std_laspx_,2), 0.0,
       0.0, pow(std_laspy_,2);

  // transform sigma points into measurement space
  // rather than re-producing sigma point in measuremnt space, it's using same sigma point in prediction step.
  double px, py;
  // cout << "Xsig_pred_ \n" << Xsig_pred_ << endl;
  for (int i=0; i<2 * n_aug_ + 1; i++){
      px = Xsig_pred_(0,i);
      py = Xsig_pred_(1,i);
      Zsig.col(i) << px,
                     py;
  }

  // cout << "Zsig: \n" << Zsig << endl;
  // calculate mean predicted measurement
  for(int j=0; j<2 * n_aug_ + 1; j++){
    z_pred += weights_(j)*Zsig.col(j);
  }
  // cout << "z_pred: \n" << z_pred << endl;

  // calculate innovation covariance matrix S
  for(int k=0; k<2 * n_aug_ + 1; k++){
    // residual
    VectorXd z_diff = Zsig.col(k) - z_pred;
    S += weights_(k)*(z_diff)*(z_diff).transpose();
  }
  S += R;
  // cout << " S: \n" <<  S << endl;

  // print result
  // std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  // std::cout << "S: " << std::endl << S << std::endl;

  // write result
  *z_pred_ = z_pred;
  *S_pred_ = S;
  *Zsig_ = Zsig;
}
void UKF::PredictRadarMeasurement(VectorXd* z_pred_, MatrixXd* S_pred_, MatrixXd *Zsig_){

  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0.0);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);

  // noise covariance matrix S
  MatrixXd R = MatrixXd(n_z,n_z);
  R << pow(std_radr_,2), 0.0, 0.0,
       0.0, pow(std_radphi_,2), 0.0,
       0.0, 0.0, pow(std_radrd_,2);

  // transform sigma points into measurement space
  // rather than re-producing sigma point in measuremnt space, it's using same sigma point in prediction step.
  double px, py, vel, yaw, yawVel;
  double 	rho, theta, rhoVel;
  // cout << "Xsig_pred_ \n" << Xsig_pred_ << endl;
  for (int i=0; i<2 * n_aug_ + 1; i++){
      px = Xsig_pred_(0,i);
      py = Xsig_pred_(1,i);
      vel = Xsig_pred_(2,i);
      yaw = Xsig_pred_(3,i);
      yawVel = Xsig_pred_(4,i);
      rho = sqrt(px*px+py*py);
      theta = atan2(py,px);
      rhoVel = (px*cos(yaw)*vel + py*sin(yaw)*vel)/(sqrt(px*px + py*py));
      Zsig.col(i) << rho,
                     theta,
                     rhoVel;
  }

  // cout << "Zsig: \n" << Zsig << endl;
  // calculate mean predicted measurement
  for(int j=0; j<2 * n_aug_ + 1; j++){
    z_pred += weights_(j)*Zsig.col(j);
  }
  // cout << "z_pred: \n" << z_pred << endl;

  // calculate innovation covariance matrix S
  for(int k=0; k<2 * n_aug_ + 1; k++){
    // residual
    VectorXd z_diff = Zsig.col(k) - z_pred;
    // angle normalization!!!
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    S += weights_(k)*(z_diff)*(z_diff).transpose();
  }
  S += R;
  // cout << " S: \n" <<  S << endl;

  // print result
  // std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  // std::cout << "S: " << std::endl << S << std::endl;

  // write result
  *z_pred_ = z_pred;
  *S_pred_ = S;
  *Zsig_ = Zsig;
}

void UKF::UpdateState(MeasurementPackage meas_package, MatrixXd Zsig_pred, VectorXd z_pred, MatrixXd S_pred, VectorXd* x_pred, MatrixXd* P_pred, std::ofstream& sensorDataFile){


  // set measurement dimension, radar can measure r, phi, and r_dot
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
    
    // cout << "=====================Update state Radar starts =====================" <<endl;
    // cout << "Zsig_pred: \n"<< Zsig_pred<< endl;
    // cout << "S_pred: \n"<< S_pred<< endl;
    // cout << "x_pred: \n"<< *x_pred<< endl;
    // cout << "P_pred: \n"<< *P_pred<< endl;

    int n_z = 3; // rho(m), phi(rad), rho_dot(m/s)
    
    // create example vector for incoming radar measurement
    VectorXd z = VectorXd(n_z); // rho(m), phi(rad), rho_dot(m/s)
    z << meas_package.raw_measurements_[0],
        meas_package.raw_measurements_[1],
        meas_package.raw_measurements_[2];
    // create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.fill(0.0);
    // calculate cross correlation matrix
    for(int i=0; i<2 * n_aug_ + 1; i++){
      VectorXd x_diff;
      VectorXd z_diff;
      // residual
      x_diff = Xsig_pred_.col(i) - x_;
      // state difference
      z_diff = Zsig_pred.col(i) - z_pred;
      while (x_diff(3)>M_PI) x_diff(3)-= 2*M_PI;
      while (x_diff(3)<-M_PI) x_diff(3)+= 2*M_PI;
      while (z_diff(1)>M_PI) z_diff(1)-= 2*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1)+= 2*M_PI;
      
      Tc += weights_(i)*x_diff*z_diff.transpose();
    }
    // cout << "z_pred: \n"<< z_pred<< endl;
    // cout << "z: \n"<< z << endl;

    // cout << "Tc: \n"<< Tc<< endl;
    // calculate Kalman gain K;
    MatrixXd K; // Kalman gain
    K.fill(0.0);
    K = Tc*S_pred.inverse();
    // cout << "K: \n"<< K << endl;

    // update state mean and covariance matrix
    // residual
    VectorXd z_diff = z - z_pred;
    while (z_diff(1)>M_PI) z_diff(1)-= 2*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+= 2*M_PI;
    x_ = x_ + K*z_diff;

    P_ = P_ - K*S_pred*K.transpose();

    // print result
    // std::cout << "Updated state x: " << std::endl << x_ << std::endl;
    // std::cout << "Updated state covariance P: " << std::endl << P_ << std::endl;

    // Normalized Innovation Squared (NIS)
    VectorXd NIS = VectorXd(1);
    VectorXd zDiff = z- z_pred;
    while (zDiff(1)>M_PI) zDiff(1)-= 2*M_PI;
    NIS = zDiff.transpose()*S_pred.inverse()*zDiff;
    sensorDataFile << NIS << ", ";
    // sensorDataFile << meas_package.sensor_type_ <<" (0:LASER, 1:RADAR) NIS: " << NIS << endl;
    // cout << "NIS: " << NIS << endl;
    // sensorDataFile << NIS << "\t";
    // write result
    *x_pred = x_;
    *P_pred = P_;
    // cout << "=====================Update state RADAR end =====================" << endl << endl;
  }
  else{ // LASER case
    // cout << "=====================Update state Lidar starts =====================" <<endl;
    // cout << "Zsig_pred: \n"<< Zsig_pred<< endl;
    // cout << "S_pred: \n"<< S_pred<< endl;
    // cout << "x_pred: \n"<< x_<< endl;
    // cout << "P_pred: \n"<< P_<< endl;

    int n_z = 2; //px(m),py(m)
    
    // create example vector for incoming radar measurement
    VectorXd z = VectorXd(n_z); //px(m),py(m)
    z << meas_package.raw_measurements_[0],
        meas_package.raw_measurements_[1];
    
    // matrix for projecting your belief about the object's current state into the measurement space of the sensor.
    MatrixXd H_ = MatrixXd(n_z,n_x_);
    H_ << 1,0,0,0,0,
          0,1,0,0,0;
    MatrixXd Ht = H_.transpose();

    // measuremnt prediction step
    VectorXd z_pred = H_ * x_;

    // update state mean and covariance matrix
    // residual
    VectorXd y = z - z_pred;
    // cout << "z_pred: \n"<< z_pred<< endl;
    // cout << "z: \n"<< z << endl;

    // measurement noise covariance matrix S
    MatrixXd R_ = MatrixXd(n_z,n_z);
    R_ << pow(std_laspx_,2), 0.0,
          0.0, pow(std_laspy_,2);
       
    MatrixXd S = H_ * P_ * Ht + R_;
    // calculate Kalman gain K
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

    // cout << "K: \n"<< K << endl;

    //new estimate
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
    
    // print result
    // std::cout << "Updated state x: " << std::endl << x_ << std::endl;
    // std::cout << "Updated state covariance P: " << std::endl << P_ << std::endl;

    // Normalized Innovation Squared (NIS)
    VectorXd NIS = VectorXd(1);
    NIS = y.transpose()*S.inverse()*y;
    sensorDataFile << NIS << ", ";
    // sensorDataFile << meas_package.sensor_type_ <<" (0:LASER, 1:RADAR) NIS: " << NIS << endl;
    // cout << "NIS: " << NIS << endl;
    // sensorDataFile << NIS << "\t";
    // write result
    *x_pred = x_;
    *P_pred = P_;
    // cout << "=====================Update state Lidar end =====================" << endl << endl;
  }
  // cout << "=====================Update state end =====================" << endl << endl;
}