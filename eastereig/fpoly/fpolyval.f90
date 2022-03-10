! This file is part of eastereig, a library to locate exceptional points
! and to reconstruct eigenvalues loci.

! Eastereig is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.

! Eastereig is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.

! You should have received a copy of the GNU General Public License
! along with Eastereig.  If not, see <https://www.gnu.org/licenses/>.


! ======================================================================
! Fortran implementation of univariate and multivariate polynomial.
! ======================================================================
! All evaluation are performed for *complex* scalar argument.
! In all cases the coefficients given in the array a
! the variable x is always the 1st dimention, y the second, and z the third 
! based on https://stackoverflow.com/questions/18707984/why-is-univariate-horner-in-fortran-faster-than-numpy-counterpart-while-bivariat


subroutine fpolyval(x, a, pval, nx)
! Evaluation of a uni-variate polynomial with convention a_i x**i, i=0, nx

    implicit none

    complex(kind=8), dimension(nx), intent(in) :: a
    complex(kind=8), intent(in) :: x
!    logical, optional, intent(in) :: tensor ! just here for interface compatibility 
    complex(kind=8), intent(out) :: pval
    integer, intent(in) :: nx
    integer :: i

    pval = 0.0d0
    do i = nx, 1, -1
        pval = pval*x + a(i)
    end do

end subroutine fpolyval


subroutine fpolyval2(x, y, a, pval, nx, ny)
! Evaluation of bi-variate polynomial with convention a_ij x**i * y**j


    implicit none

    complex(kind=8), dimension(nx, ny), intent(in) :: a
    complex(kind=8), intent(in) :: x, y
    complex(kind=8), intent(out) :: pval
    integer, intent(in) :: nx, ny
    complex(kind=8) :: tmp
    integer :: i, j

    pval = 0.d0
    do j = ny, 1, -1
        tmp = 0.d0
        do i = nx, 1, -1
        	! inner loop on the row (fortran  layout)
            tmp = tmp*x + a(i, j)
        end do
        pval = pval*y + tmp
    end do

end subroutine fpolyval2


subroutine fpolyval3(x, y, z, a, pval, nx, ny, nz)
! Evaluation of a tri-variate polynomial with convention a_ijk x**i * y**j * z**k

    implicit none

    complex(8), dimension(nx, ny, nz), intent(in) :: a
    complex(8), intent(in) :: x, y, z
    complex(8), intent(out) :: pval
    integer, intent(in) :: nx, ny, nz
    complex(8) :: tmp, tmp2
    integer :: i, j, k

    pval = 0.0d0
    do k = nz, 1, -1
        tmp2 = 0.0d0
        do j = ny, 1, -1
            tmp = 0.0d0
            do i = nx, 1, -1
                ! inner loop on the row (fortran  layout)
                tmp = tmp*x + a(i, j, k)
            end do
            tmp2 = tmp2*y + tmp
        end do
        pval = pval*z + tmp2
    end do

end subroutine fpolyval3


subroutine fpolyval4(x, y, z, zz, a, pval, nx, ny, nz, nzz)
! Evaluation of a quadri-variate polynomial with convention a_ijkl x**i * y**j * z**k * zz**l

    implicit none

    complex(8), dimension(nx, ny, nz, nzz), intent(in) :: a
    complex(8), intent(in) :: x, y, z, zz
    complex(8), intent(out) :: pval
    integer, intent(in) :: nx, ny, nz, nzz
    complex(8) :: tmp, tmp2, tmp3
    integer :: i, j, k, l

    pval = 0.0d0
    do l = nzz, 1, -1
        tmp3 = 0.0d0
        do k = nz, 1, -1
            tmp2 = 0.0d0
            do j = ny, 1, -1
                tmp = 0.0d0
                do i = nx, 1, -1
                    ! inner loop on the row (fortran  layout)
                    tmp = tmp*x + a(i, j, k, l)
                end do
                tmp2 = tmp2*y + tmp
            end do
            tmp3 = tmp3*z + tmp2
        end do
        pval = pval*zz + tmp3
    end do

end subroutine fpolyval4
