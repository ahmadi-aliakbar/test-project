% Transforms the components of a vector (linear position, velocity, acceleration, etc.) 
% from the Body-fixed coordinate system to the North-East-Down (NED) vehicle carried 
% coordinate system using Euler angles Roll (ph), Pitch (th), and Yaw (ps), implementing
% a 3-2-1 rotation sequence.

% Inputs: 
%Euler angles (1x3 or 3x1: roll, pitch, and yaw (in that order) [rad].
%   Body-fixed linear vector components to transform, e.g., U, V, W
%     U - Pointing forward, lying in the symmetric plane of the flying vehicle
%     V - Pointing to the right side of the flying vehicle
%     W - Pointing downwards, completing the right-hand rule

% Outputs:
%   Linear vector components in the vehicle-carried North-East-Down coordinate system, 
%   as a 3x1 vector with components:
%     X - Pointing North
%     Y - Pointing East
%     Z - Pointing Down

% Raul A. Garcia-Huerta, ragarcia@iteso.mx, 2019-06-21.
% Code created using MATLAB R2020a v9.8.

function LinVecTrasformed = body2ned(EulerAngles, BodyLinVec)

  ph = EulerAngles(1);
  th = EulerAngles(2);
  ps = EulerAngles(3);

  Rnvb = [cos(th)*cos(ps)  sin(ph)*sin(th)*cos(ps)-cos(ph)*sin(ps)  cos(ph)*sin(th)*cos(ps)+sin(ph)*sin(ps) ;...
          cos(th)*sin(ps)  sin(ph)*sin(th)*sin(ps)+cos(ph)*cos(ps)  cos(ph)*sin(th)*sin(ps)-sin(ph)*cos(ps) ;...
              -sin(th)                     sin(ph)*cos(th)                          cos(ph)*cos(th)         ];

  LinVecTrasformed = Rnvb*[BodyLinVec(1); BodyLinVec(2); BodyLinVec(3)];

end