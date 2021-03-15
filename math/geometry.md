# 几何

## 二维

- 点： $(x_0, y_0)$
- 直线：$Ax + By + C = 0$
  - 方向向量：$(B, -A)$
  - 法向量：$(-A, B)$
  - 两平行直线之间的距离：$\cfrac{|C_1 - C_2|}{\sqrt{A^2 + B^2}}$
- 点到直线的距离：$d = \cfrac{|Ax_0 + By_0 + C|}{\sqrt{A^2 + B^2}}$

## 三维

- 点：$(x_0, y_0, z_0)$
- 平面：$Ax + By + Cz + D = 0$
  - 法向量：$(A, B, C)$
  - 两平行平面之间的距离：$\cfrac{|D_1 - D_2|}{\sqrt{A^2 + B^2 + C^2}}$
- 点到平面的距离：$d = \cfrac{|Ax_0 + By_0 + Cz_0 + D|}{\sqrt{A^2 + B^2 + C^2}}$

## 超平面

- 点：$\boldsymbol{x_0}$
- 平面：$\boldsymbol{w}^\mathrm{T} \boldsymbol{x} + b = 0$
  - 法向量：$\boldsymbol{w}$
  - 截距：$b$
  - 两平行超平面之间的距离：$\cfrac{|b_1 - b_2|}{\|w\|}$
- 点到平面的距离：$d = \cfrac{|\boldsymbol{w}^\mathrm{T} \boldsymbol{x_0} + b|}{\|\boldsymbol{w}\|}$
