import cupy as cp
import numpy as np


def _eigh_gram(A):
    """Eigendecompose A @ A.T efficiently.
    When A is (d, k) with d > k, works with the smaller (k, k) matrix instead.
    Returns (eigenvalues_descending, eigenvectors)."""
    d, k = A.shape
    if d <= k:
        G = A @ cp.transpose(A)
        vals, vecs = cp.linalg.eigh(G)
        return vals[::-1], vecs[:, ::-1]
    M = cp.transpose(A) @ A
    vals, V = cp.linalg.eigh(M)
    vals = vals[::-1]
    V = V[:, ::-1]
    safe = vals > 1e-10
    U = cp.zeros((d, k), dtype=A.dtype)
    if cp.any(safe):
        U[:, safe] = A @ V[:, safe] / cp.sqrt(vals[safe])
    return vals, U


def _eigh_proj_diff(basis, overlap):
    """Eigendecompose basis@basis.T - overlap@overlap.T efficiently.
    Assumes basis columns are orthonormal and overlap in span(basis).
    Works in the r-dimensional coordinate space of basis.
    Returns (eigenvalues_descending, eigenvectors)."""
    r = basis.shape[1]
    BtO = cp.transpose(basis) @ overlap
    M = cp.eye(r, dtype=basis.dtype) - BtO @ cp.transpose(BtO)
    vals, v = cp.linalg.eigh(M)
    vals = vals[::-1]
    v = v[:, ::-1]
    vecs = basis @ v
    return vals, vecs


def _get_array_module(*arrays):
    """전달된 배열들이 cupy 배열이면 cupy 모듈을, 아니면 numpy를 반환."""
    try:
        import cupy as cp
        for a in arrays:
            if isinstance(a, cp.ndarray):
                return cp
    except ImportError:
        pass
    return np

class SubspaceDiff():
    def __init__(self):
        self.cgTool = CompleteProjection()
        pass

    def calc_sum_subspace(self, basis1, basis2):

        C = cp.concatenate((basis1, basis2), axis=1)
        alphas, lambdas = _eigh_gram(C)
        alphas, lambdas = self.adjust_eig(alphas, lambdas)

        index = 0
        for j, element in enumerate(alphas):
            if element < 0.0:
                index = j
                break

        return [alphas[0:index], lambdas[:,0:index]]

    def calc_overlap_subspace(self, basis1, basis2, isVerbose=False):

        C = cp.concatenate((basis1, basis2), axis=1)
        alphas, lambdas = _eigh_gram(C)

        index = 0
        for i, a in enumerate(alphas):
            if a < 2.0 - 1e-4:
                index = i
                break

        if isVerbose: print("{}-dimension is overlapped!".format(index))

        return [alphas[0:index], lambdas[:,0:index]]
    
    def calc_karcher_subspace(self, basis1, basis2):

        if basis1.shape[1] == 1 and basis2.shape[1] == 1:
            result = (basis1 + basis2) / 2
            result = result / cp.linalg.norm(result)
            return [[1], result]

        C = cp.concatenate((basis1, basis2), axis=1)
        alphas, lambdas = _eigh_gram(C)
        alphas, lambdas = self.adjust_eig(alphas, lambdas)

        index = 0
        for i, a in enumerate(alphas):
            if a < 1.0:
                index = i
                break

        return [alphas[0:index], lambdas[:,0:index]]
    
    def calc_diff_subspace(self, basis1, basis2):

        C = cp.concatenate((basis1, basis2), axis=1)
        alphas, lambdas = _eigh_gram(C)
        alphas, lambdas = self.adjust_eig(alphas, lambdas)
        # print(alphas)

        index_start = 0
        for i, a in enumerate(alphas):
            if a < 1.0:
                index_start = i
                break

        index_end = 0
        for i, a in enumerate(alphas):
            if a < 1e-5:
                index_end = i
                break

        print("dim : {}".format(index_end - index_start))
        # print(index_start)
        # print(index_end)

        return [alphas[index_start:index_end], lambdas[:,index_start:index_end]]
    
    def adjust_eig(self, eigenvalues, eigenvectors, epsilon=1e-9):
        eig_num = eigenvalues.shape[0]

        eigenvalues_new = []
        eigenvectors_new = []

        for i in range(eig_num):
            value = eigenvalues[i]
            if value >= 2.0 or (value < 2.0 and value > 2.0 - epsilon): # dims overlapped
                eigenvalues_new.append(2.0)
                eigenvectors_new.append(eigenvectors[:,i])
            elif (value < epsilon and value > 0.0) or value <= 0.0: # dims cannot express
                eigenvalues_new.append(0.0)
                eigenvectors_new.append(eigenvectors[:,i])
            else:
                eigenvalues_new.append(value.get())
                eigenvectors_new.append(eigenvectors[:,i])

        return [cp.array(eigenvalues_new), cp.transpose(cp.array(eigenvectors_new))]

    def calc_magnitude(self, basis1, basis2, isVerbose=False, overlapFlag=True):
        
        if not overlapFlag:

            _, basis_overlap = self.calc_overlap_subspace(basis1, basis2, isVerbose=isVerbose)

            alpha1, basis1_revised = _eigh_proj_diff(basis1, basis_overlap)
            index_1 = 0
            for i, a in enumerate(alpha1):
                if abs(a) < 1e-1 :
                    index_1 = i
                    break
            basis1_revised = basis1_revised[:,:index_1]

            alpha2, basis2_revised = _eigh_proj_diff(basis2, basis_overlap)
            index_2 = 0
            for j, a in enumerate(alpha2):
                if abs(a) < 1e-1 :
                    index_2 = j
                    break
            basis2_revised = basis2_revised[:,:index_2]

        else:
            basis1_revised = basis1
            basis2_revised = basis2
        # print("index1 : {}".format(index_1))
        # print(alpha1)
        # print("index2 : {}".format(index_2))
        # print(alpha2)

        G = cp.transpose(basis1_revised)@basis2_revised
        _, s, _ = cp.linalg.svd(G)
        if isVerbose:
            for ele_s in s:
                print("{:.2f}, ".format(ele_s), end="")
            print()

        return 2 * (len(s) - cp.sum(s))
    
    def calc_similarity(self, basis1, basis2, isVerbose=False):
        
        _, basis_overlap = self.calc_overlap_subspace(basis1, basis2, isVerbose=isVerbose)
        tmp = basis_overlap.shape[1]

        alpha1, basis1_revised = _eigh_proj_diff(basis1, basis_overlap)
        index_1 = 0
        for i, a in enumerate(alpha1):
            if abs(a) < 1- 1e-4 :
                index_1 = i
                break
        basis1_revised = basis1_revised[:,:index_1]

        alpha2, basis2_revised = _eigh_proj_diff(basis2, basis_overlap)
        index_2 = 0
        for j, a in enumerate(alpha2):
            if abs(a) < 1- 1e-4 :
                index_2 = j
                break
        basis2_revised = basis2_revised[:,:index_2]

        G = cp.transpose(basis1_revised)@basis2_revised
        _, s, _ = cp.linalg.svd(G)
        if isVerbose: print(s)

        return cp.sum(s) + tmp
    
    def calc_1st_magnitude_decomposed(self, basis1, basis2, basis3):

        W_tmp = cp.concatenate((basis1, basis3), axis=1)
        l, W = _eigh_gram(W_tmp)
        idx = 0
        for i, ele_l in enumerate(l):
            if ele_l < 1e-4:
                idx = i
                break
        W = W[:,:idx]
        _, s, _ = cp.linalg.svd(cp.transpose(W)@basis2)
        U, _ = cp.linalg.qr(cp.transpose(W)@basis2)
        basis2_prime = W@U

        mag_orth = 2 * (len(s) - cp.sum(s))
        mag_along = self.calc_magnitude(basis2_prime, basis1)

        return [mag_along, mag_orth]

    def calc_2nd_magnitude_decomposed(self, basis1, basis2, basis3):

        # if (len(basis1.shape) == 1) :
        #     new_basis1 = cp.expand_dims(basis1, axis=-1)
        #     new_basis2 = cp.expand_dims(basis2, axis=-1)
        #     new_basis3 = cp.expand_dims(basis3, axis=-1)
        # else:
        #     new_basis1 = basis1
        #     new_basis2 = basis2
        #     new_basis3 = basis3

        W_tmp = cp.concatenate((basis1, basis3), axis=1)
        W, _ = cp.linalg.qr(W_tmp)
        _, M = self.calc_karcher_subspace(basis1, basis3)
        # _, s, _ = cp.linalg.svd(cp.transpose(W)@basis2)
        U, _ = cp.linalg.qr(cp.transpose(W)@basis2)
        basis2_prime = W@U
        _, s, _ = cp.linalg.svd(cp.transpose(basis2_prime)@basis2)

        # Gt = self.cgTool.geodesic_between(basis1, basis3)
        # _, M = self.calc_karcher_subspace(basis1, basis3)
        # Ts = cp.arange(-10.0, 30.0 + 1e-9, 1e-4)
        # basis2_star, _, t = self.cgTool.proj2geo_fminbnd_brent(basis2, Gt, Ts)
        # _, _, t_prime = self.cgTool.proj2geo_fminbnd_brent(basis2_prime, Gt, Ts)

        # print("1 and 2 : {:.4f}".format(self.cgTool.grassmann_dist(basis1, basis2)), end=" || ")
        # print("2 and 3 : {:.4f}".format(self.cgTool.grassmann_dist(basis2, basis3)), end=" || ")
        # print("1 and 3 : {:.4f}".format(self.cgTool.grassmann_dist(basis1, basis3)), end=" || ")
        # print("2 and approx : {:.4f}".format(self.cgTool.grassmann_dist(basis2, basis2_prime)), end=" || ")
        # print("2 and truth : {:.4f}".format(self.cgTool.grassmann_dist(basis2, basis2_star)), end=" || ")
        # print("approx and truth : {:.4f}".format(self.cgTool.grassmann_dist(basis2_star, basis2_prime)))
        # print("t : {} / len : {}".format(t, len(Ts)))
        # print("t_prime : {} / len : {}".format(t_prime, len(Ts)))

        mag_orth = 2 * (len(s) - cp.sum(s))
        # mag_orth = cp.sum(s)
        mag_along = self.calc_magnitude(basis2_prime, M)
        # mag_along = self.calc_similarity(basis2_prime, M)

        return [mag_along, mag_orth]

    def calc_2nd_magnitude_decomposed_real(self, basis1, basis2, basis3):

        # if (len(basis1.shape) == 1) :
        #     new_basis1 = cp.expand_dims(basis1, axis=-1)
        #     new_basis2 = cp.expand_dims(basis2, axis=-1)
        #     new_basis3 = cp.expand_dims(basis3, axis=-1)
        # else:
        #     new_basis1 = basis1
        #     new_basis2 = basis2
        #     new_basis3 = basis3

        Gt = self.cgTool.geodesic_between(basis1, basis3)
        _, M = self.calc_karcher_subspace(basis1, basis3)
        Ts = cp.arange(-10.0, 30.0 + 1e-9, 1e-4)
        basis2_star, _, t = self.cgTool.proj2geo_fminbnd_brent(basis2, Gt, Ts)

        W_tmp = cp.concatenate((basis1, basis3), axis=1)
        W, _ = cp.linalg.qr(W_tmp)
        # _, M = self.calc_karcher_subspace(basis1, basis3)
        # _, s, _ = cp.linalg.svd(cp.transpose(W)@basis2)
        U, _ = cp.linalg.qr(cp.transpose(W)@basis2)
        basis2_prime = W@U
        _, s, _ = cp.linalg.svd(cp.transpose(basis2_prime)@basis2)
        _, _, t_prime = self.cgTool.proj2geo_fminbnd_brent(basis2_prime, Gt, Ts)

        # print("1 and 2 : {:.4f}".format(self.cgTool.grassmann_dist(basis1, basis2)), end=" || ")
        # print("2 and 3 : {:.4f}".format(self.cgTool.grassmann_dist(basis2, basis3)), end=" || ")
        # print("1 and 3 : {:.4f}".format(self.cgTool.grassmann_dist(basis1, basis3)), end=" || ")
        # print("2 and approx : {:.4f}".format(self.cgTool.grassmann_dist(basis2, basis2_prime)), end=" || ")
        # print("2 and truth : {:.4f}".format(self.cgTool.grassmann_dist(basis2, basis2_star)), end=" || ")
        # print("approx and truth : {:.4f}".format(self.cgTool.grassmann_dist(basis2_star, basis2_prime)))
        # print("t : {} / len : {}".format(t, len(Ts)))
        # print("t_prime : {} / len : {}".format(t_prime, len(Ts)))

        G = cp.transpose(basis2)@basis2_star
        _, s, _ = cp.linalg.svd(G)

        mag_orth = 2 * (len(s) - cp.sum(s))
        mag_along = self.calc_magnitude(basis2_star, M)

        return [mag_along, mag_orth]
    
    def calc_rbf_magnitude(self, alphas1, alphas2, km):
        _, s, _ = cp.linalg.svd(cp.transpose(alphas2) @ cp.transpose(km) @ alphas1)
        return 2 * (len(s) - cp.sum(s))
    
class Grassmannian():
    def __init__(self):
        pass
    
    def get_grassmannian(self, basis):
        P = basis@cp.transpose(basis)
        return P
    
    def get_symmetric(self, X):
        symX = 0.5 * (X + cp.transpose(X))
        return symX
    
    def get_tangent_projection(self, X, D):
        I = cp.eye(X.shape[0])
        tmp = X @ self.get_symmetric(D) @ (I - X)
        return 2 * self.get_symmetric(tmp)
    
    def calc_grassmannian_inner_product(self, zeta, eta):
        return cp.trace(cp.transpose(zeta)@eta)
        
    def calc_grassmannian_norm(self, zeta):
        return cp.sqrt(cp.trace(cp.transpose(zeta)@zeta))
    
    def normalize(self, X):
        return X / self.calc_grassmannian_norm(X)
    
    def logarithmic_mapping(self, K, X):

        YTU = cp.transpose(X)@K

        _, Q_tilde = cp.linalg.eigh(YTU@cp.transpose(YTU))
        Q_tilde = Q_tilde[:,::-1]
        _, R_tilde = cp.linalg.eigh(cp.transpose(YTU)@YTU)
        R_tilde = R_tilde[:,::-1]
        Y_prime = X @ Q_tilde @ cp.transpose(R_tilde)

        result = (cp.eye(K.shape[0]) - K@cp.transpose(K)) @ Y_prime
        _, Q = _eigh_gram(result)
        _, R = cp.linalg.eigh(cp.transpose(result)@result)
        R = R[:,::-1]
        _, sigma, _ = cp.linalg.svd(result)

        return Q[:,:sigma.shape[0]] @ cp.diag(cp.arcsin(sigma)) @ cp.transpose(R)
    
"""
proj2geo_fminbnd — NumPy / CuPy 양쪽에서 동작하는 버전.

사용법:
    import numpy as np
    Gt = geodesic_fn(XV, XpU, Th)              # XV, XpU, Th 가 numpy면 CPU
    Z_star, t_star, idx = proj2geo_fminbnd(Z, Gt, Ts)

    import cupy as cp
    Gt = geodesic_fn(XV_gpu, XpU_gpu, Th_gpu)  # cupy 배열이면 GPU
    Z_star, t_star, idx = proj2geo_fminbnd(Z_gpu, Gt, Ts_gpu)

설계 메모
---------
1차원 최적화의 변수 t 는 항상 Python float (CPU 스칼라) 입니다.
GPU 에서 도는 것은 Gt(t) 와 grassmann_dist 의 행렬 연산뿐입니다.
이 덕분에 scipy.optimize.minimize_scalar 도 그대로 쓸 수 있고,
원한다면 자체 Brent 구현으로도 전환할 수 있습니다.
"""

class CompleteProjection():

    # =================================================
    # grassmann_dist
    # =================================================
    def grassmann_dist(self, X, Y):
        """
        두 부분공간의 그라스만 거리 (주각의 L2 norm).

        X, Y: (d, r) ndarray (numpy 또는 cupy), 열들이 정규직교.
        반환값: Python float.
        """
        xp = _get_array_module(X, Y)
        s = xp.linalg.svd(X.T @ Y, compute_uv=False)
        s = xp.clip(s, -1.0, 1.0)
        th = xp.arccos(s)
        d = xp.linalg.norm(th)
        # 스칼라로 강제 변환 (cupy 0-d 배열도 float() 가능)
        return float(d)
    
    def grassmann_log(self, X, S):
        """
        그라스만 로그 사상: X 의 접공간 위에서 S 를 가리키는 접벡터의
        SVD 인자 (U_tilde, Th, V) 를 반환.
    
        Delta = X_perp * U_tilde * Th * V^T 가 X 에서 S 로 가는 접벡터.
    
        Parameters
        ----------
        X, S : (d, r) ndarray
            정규직교 열을 가진 부분공간 표현.
    
        Returns
        -------
        U_tilde : (d, r) ndarray
            X 의 직교여공간 안의 좌특이벡터 (정규직교).
        Th      : (r, r) ndarray
            주각 diag(theta_1, ..., theta_r).
        V       : (r, r) ndarray
            우특이벡터 (X 기저를 정렬하는 회전).
        """
        xp = _get_array_module(X, S)
        d, r = X.shape
    
        # 보조 행렬 M = (I - X X^T) S (X^T S)^{-1}
        XtS = X.T @ S                                  # (r, r)
        S_perp = S - X @ XtS                           # X 에 수직 성분  (d, r)
        # (X^T S) 가 특이에 가까우면 X 와 S 가 거의 직교한다는 뜻 (theta ≈ pi/2)
        M = xp.linalg.solve(XtS.T, S_perp.T).T         # = S_perp @ inv(XtS)
    
        # M = U_tilde * tan(Th) * V^T
        U_tilde, sigma, Vt = xp.linalg.svd(M, full_matrices=False)
        V = Vt.T
    
        # 특이값이 tan(theta) 이므로 arctan 으로 주각 복원
        theta = xp.arctan(sigma)
        Th = xp.diag(theta)
    
        return U_tilde, Th, V
    
    def geodesic_between(self, X, S):
        """
        X 와 S 를 잇는 그라스만 측지선을 함수로 반환.
    
        G(0) = X (와 같은 부분공간), G(1) = S (와 같은 부분공간).
        """
        U_tilde, Th, V = self.grassmann_log(X, S)
        XV = X @ V
        XpU = U_tilde   # 이미 X 의 직교여공간 안에 있고 정규직교
        return self.geodesic_fn(XV, XpU, Th)


    # =================================================
    # geodesic_fn
    # =================================================
    def geodesic_fn(self, XV, XpU, Th):
        """
        그라스만 측지선 G(t) = XV*cos(Th*t) + XpU*sin(Th*t) 을 함수로 반환.

        XV, XpU, Th 가 cupy 배열이면 반환된 Gt 도 cupy 배열을 만든다.
        t 는 항상 Python 스칼라.
        """
        xp = _get_array_module(XV, XpU, Th)
        th_diag = xp.diag(Th)  # (r,) — XV/XpU와 같은 모듈

        def Gt(t):
            # t 는 Python 스칼라; 결과는 XV 와 같은 디바이스의 (d, r) 배열
            cos_part = XV * xp.cos(th_diag * t)
            sin_part = XpU * xp.sin(th_diag * t)
            return cos_part + sin_part

        return Gt


    # =================================================
    # proj2geo_fminbnd_brent — SciPy 의존성 제거 버전
    # =================================================
    def proj2geo_fminbnd_brent(self, Z, Gt, Ts, xatol=1e-15, maxiter=500):
        """
        proj2geo_fminbnd 와 동일하지만 Brent 알고리즘을 직접 구현.
        SciPy 가 없는 환경 (예: 순수 cupy 배포) 에서도 동작.

        구현은 Brent (1973) 의 황금분할 + 역포물선 보간 결합 알고리즘으로,
        SciPy fminbnd 와 동일한 계열이다.
        """
        xp = _get_array_module(Z, Ts)

        a = float(Ts[0])
        b = float(Ts[-1])

        def f(t):
            return self.grassmann_dist(Z, Gt(t))

        # --- Brent's bounded minimization (Forsythe/Malcolm/Moler 스타일) ---
        sqrt_eps = np.sqrt(np.finfo(float).eps)
        golden_ratio = 0.5 * (3.0 - np.sqrt(5.0))  # ≈ 0.3819660

        x = w = v = a + golden_ratio * (b - a)
        fx = fw = fv = f(x)
        d_step = e_step = 0.0

        for _ in range(maxiter):
            m = 0.5 * (a + b)
            tol1 = sqrt_eps * abs(x) + xatol
            tol2 = 2.0 * tol1
            if abs(x - m) <= tol2 - 0.5 * (b - a):
                break

            # 역포물선 보간 시도
            use_golden = True
            if abs(e_step) > tol1:
                r = (x - w) * (fx - fv)
                q = (x - v) * (fx - fw)
                p = (x - v) * q - (x - w) * r
                q = 2.0 * (q - r)
                if q > 0.0:
                    p = -p
                q = abs(q)
                e_prev = e_step
                e_step = d_step
                # 보간이 유효한지 검사
                if (abs(p) < abs(0.5 * q * e_prev)
                        and p > q * (a - x)
                        and p < q * (b - x)):
                    d_step = p / q
                    u = x + d_step
                    # 끝점에 너무 가까우면 살짝 안쪽으로
                    if (u - a) < tol2 or (b - u) < tol2:
                        d_step = tol1 if x < m else -tol1
                    use_golden = False

            if use_golden:
                e_step = (b - x) if x < m else (a - x)
                d_step = golden_ratio * e_step

            # 스텝이 tol1 보다 작아지면 tol1 만큼만 움직임
            if abs(d_step) >= tol1:
                u = x + d_step
            else:
                u = x + (tol1 if d_step > 0 else -tol1)

            fu = f(u)

            # bracket 갱신
            if fu <= fx:
                if u < x:
                    b = x
                else:
                    a = x
                v, w, x = w, x, u
                fv, fw, fx = fw, fx, fu
            else:
                if u < x:
                    a = u
                else:
                    b = u
                if fu <= fw or w == x:
                    v, w = w, u
                    fv, fw = fw, fu
                elif fu <= fv or v == x or v == w:
                    v = u
                    fv = fu

        t_star = float(x)
        Z_star = Gt(t_star)
        t_star_idx = int(xp.argmin(xp.abs(Ts - t_star)))
        return Z_star, t_star, t_star_idx


# =================================================
# 동작 확인
# =================================================
# def _run_test(xp, label):
#     print(f"--- {label} ---")
#     rng = np.random.default_rng(0)
#     d, r = 8, 2

#     # numpy 로 먼저 만든 뒤 필요하면 GPU 로 옮김
#     X_np, _ = np.linalg.qr(rng.standard_normal((d, r)))

#     Q_full, _ = np.linalg.qr(rng.standard_normal((d, d)))
#     Xp_np = Q_full - X_np @ (X_np.T @ Q_full)
#     Xp_np, _ = np.linalg.qr(Xp_np)
#     Xp_np = Xp_np[:, :d - r]

#     A = rng.standard_normal((d - r, r))
#     U_tilde_np, th_vals_np, Vt_np = np.linalg.svd(A, full_matrices=False)
#     th_vals_np = th_vals_np * 0.5
#     Th_np = np.diag(th_vals_np)
#     V_np = Vt_np.T

#     XV_np = X_np @ V_np
#     XpU_np = Xp_np @ U_tilde_np

#     # 선택된 모듈로 변환
#     XV = xp.asarray(XV_np)
#     XpU = xp.asarray(XpU_np)
#     Th = xp.asarray(Th_np)

#     Gt = geodesic_fn(XV, XpU, Th)
#     G0 = Gt(0.0)
#     print("  Gt(0)^T Gt(0) ≈ I?",
#           bool(xp.allclose(G0.T @ G0, xp.eye(r), atol=1e-10)))

#     Z = Gt(0.37)
#     Ts = xp.arange(-1.0, 3.0 + 1e-9, 1e-4)

#     # SciPy 버전
#     Z_star, t_star, idx = proj2geo_fminbnd(Z, Gt, Ts)
#     print(f"  [scipy] t_star = {t_star:.12f}, "
#           f"dist = {grassmann_dist(Z, Z_star):.3e}")

#     # 자체 Brent 버전
#     Z_star2, t_star2, idx2 = proj2geo_fminbnd_brent(Z, Gt, Ts)
#     print(f"  [brent] t_star = {t_star2:.12f}, "
#           f"dist = {grassmann_dist(Z, Z_star2):.3e}")


# if __name__ == "__main__":
#     _run_test(np, "NumPy (CPU)")
#     try:
#         import cupy as cp
#         _run_test(cp, "CuPy (GPU)")
#     except ImportError:
#         print("--- CuPy 미설치: GPU 테스트 건너뜀 ---")