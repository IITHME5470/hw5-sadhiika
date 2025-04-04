#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void grid(int nx, int nxglob, int istglob, int ienglob, double xstglob, double xenglob, double *x, double *dx) {
    int i, iglob;
    *dx = (xenglob - xstglob)/(double)(nxglob-1);
    
    for(i=0; i<nx; i++) {
        iglob = istglob + i;
        x[i] = xstglob + ((double)iglob)*(*dx);
    }
}

int main(int argc, char *argv[]) {
    int nx, ny, nxglob, nyglob;
    double xst, xen, yst, yen;
    double dt, tend, alpha, beta;
    double dx, dy;
    double *x, *y;
    double *T, *Tnew;
    double *xleftghost, *xrightghost, *ybotghost, *ytopghost;
    double t;
    int i, j, istep, nsteps;
    FILE *fout;
    char fname[256];
    
    int rank, size, dims[2], periods[2], coords[2];
    int left, right, bottom, top;
    MPI_Comm cart_comm;
    MPI_Status status;
    double start_time, end_time;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Read input parameters
    if (rank == 0) {
        FILE *fin = fopen("input2d.in", "r");
        if (fin == NULL) {
            printf("Cannot open input file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        fscanf(fin, "%d %d", &nxglob, &nyglob);
        fscanf(fin, "%lf %lf %lf %lf", &xst, &xen, &yst, &yen);
        fscanf(fin, "%lf %lf %lf %lf", &t, &dt, &alpha, &beta);
        fscanf(fin, "%lf", &tend);
        fscanf(fin, "%d %d", &dims[0], &dims[1]);
        fclose(fin);
    }
    
    // Broadcast input parameters
    MPI_Bcast(&nxglob, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nyglob, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&xst, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&xen, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&yst, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&yen, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tend, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Create 2D Cartesian topology
    periods[0] = 0;
    periods[1] = 0;
    
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    
    // Find neighboring processes
    MPI_Cart_shift(cart_comm, 0, 1, &left, &right);
    MPI_Cart_shift(cart_comm, 1, 1, &bottom, &top);
    
    // Calculate local grid dimensions
    nx = nxglob / dims[0];
    ny = nyglob / dims[1];
    
    // Adjust for remainder
    if (coords[0] < nxglob % dims[0]) nx++;
    if (coords[1] < nyglob % dims[1]) ny++;
    
    // Calculate global indices
    int istglob = 0;
    int jstglob = 0;
    
    // Calculate starting global indices for this process
    for (i = 0; i < coords[0]; i++) {
        istglob += nxglob / dims[0];
        if (i < nxglob % dims[0]) istglob++;
    }
    
    for (j = 0; j < coords[1]; j++) {
        jstglob += nyglob / dims[1];
        if (j < nyglob % dims[1]) jstglob++;
    }
    
    int ienglob = istglob + nx - 1;
    int jenglob = jstglob + ny - 1;
    
    // Allocate memory for arrays
    x = (double*)malloc((nx+2) * sizeof(double));
    y = (double*)malloc((ny+2) * sizeof(double));
    T = (double*)malloc((nx+2) * (ny+2) * sizeof(double));
    Tnew = (double*)malloc((nx+2) * (ny+2) * sizeof(double));
    
    // Ghost arrays for halo exchange
    xleftghost = (double*)malloc(ny * sizeof(double));
    xrightghost = (double*)malloc(ny * sizeof(double));
    ybotghost = (double*)malloc(nx * sizeof(double));
    ytopghost = (double*)malloc(nx * sizeof(double));
    
    // Generate grid
    grid(nx, nxglob, istglob, ienglob, xst, xen, &x[1], &dx);
    grid(ny, nyglob, jstglob, jenglob, yst, yen, &y[1], &dy);
    
    // Initialize temperature field with proper initial condition
    for (j = 1; j <= ny; j++) {
        for (i = 1; i <= nx; i++) {
            // Initial condition: T = 0 everywhere except at boundaries
            if (istglob + i - 1 == 0 || istglob + i - 1 == nxglob - 1 || 
                jstglob + j - 1 == 0 || jstglob + j - 1 == nyglob - 1) {
                T[j*(nx+2) + i] = 1.0; // Boundary condition T = 1
            } else {
                T[j*(nx+2) + i] = 0.0;
            }
        }
    }
    
    // Calculate number of time steps
    nsteps = 10; // Only run 10 steps for comparison
    
    // Time stepping loop
    for (istep = 0; istep < nsteps; istep++) {
        // Perform halo exchange for each time step
        // Send to right, receive from left
        if (right != MPI_PROC_NULL) {
            for (j = 1; j <= ny; j++) {
                xrightghost[j-1] = T[j*(nx+2) + nx];
            }
            MPI_Send(xrightghost, ny, MPI_DOUBLE, right, 0, cart_comm);
        }
        
        if (left != MPI_PROC_NULL) {
            MPI_Recv(xleftghost, ny, MPI_DOUBLE, left, 0, cart_comm, &status);
            for (j = 1; j <= ny; j++) {
                T[j*(nx+2) + 0] = xleftghost[j-1];
            }
        }
        
        // Send to left, receive from right
        if (left != MPI_PROC_NULL) {
            for (j = 1; j <= ny; j++) {
                xleftghost[j-1] = T[j*(nx+2) + 1];
            }
            MPI_Send(xleftghost, ny, MPI_DOUBLE, left, 1, cart_comm);
        }
        
        if (right != MPI_PROC_NULL) {
            MPI_Recv(xrightghost, ny, MPI_DOUBLE, right, 1, cart_comm, &status);
            for (j = 1; j <= ny; j++) {
                T[j*(nx+2) + nx+1] = xrightghost[j-1];
            }
        }
        
        // Send to top, receive from bottom
        if (top != MPI_PROC_NULL) {
            for (i = 1; i <= nx; i++) {
                ytopghost[i-1] = T[ny*(nx+2) + i];
            }
            MPI_Send(ytopghost, nx, MPI_DOUBLE, top, 2, cart_comm);
        }
        
        if (bottom != MPI_PROC_NULL) {
            MPI_Recv(ybotghost, nx, MPI_DOUBLE, bottom, 2, cart_comm, &status);
            for (i = 1; i <= nx; i++) {
                T[0*(nx+2) + i] = ybotghost[i-1];
            }
        }
        
        // Send to bottom, receive from top
        if (bottom != MPI_PROC_NULL) {
            for (i = 1; i <= nx; i++) {
                ybotghost[i-1] = T[1*(nx+2) + i];
            }
            MPI_Send(ybotghost, nx, MPI_DOUBLE, bottom, 3, cart_comm);
        }
        
        if (top != MPI_PROC_NULL) {
            MPI_Recv(ytopghost, nx, MPI_DOUBLE, top, 3, cart_comm, &status);
            for (i = 1; i <= nx; i++) {
                T[(ny+1)*(nx+2) + i] = ytopghost[i-1];
            }
        }
        
        // Apply boundary conditions
        if (left == MPI_PROC_NULL) {
            for (j = 1; j <= ny; j++) {
                T[j*(nx+2) + 0] = 1.0; // Left boundary
            }
        }
        
        if (right == MPI_PROC_NULL) {
            for (j = 1; j <= ny; j++) {
                T[j*(nx+2) + nx+1] = 1.0; // Right boundary
            }
        }
        
        if (bottom == MPI_PROC_NULL) {
            for (i = 1; i <= nx; i++) {
                T[0*(nx+2) + i] = 1.0; // Bottom boundary
            }
        }
        
        if (top == MPI_PROC_NULL) {
            for (i = 1; i <= nx; i++) {
                T[(ny+1)*(nx+2) + i] = 1.0; // Top boundary
            }
        }
        
        // Compute new temperature using explicit Euler scheme
        for (j = 1; j <= ny; j++) {
            for (i = 1; i <= nx; i++) {
                // Skip boundary nodes (they are fixed)
                if ((istglob + i - 1 == 0 || istglob + i - 1 == nxglob - 1 || 
                     jstglob + j - 1 == 0 || jstglob + j - 1 == nyglob - 1)) {
                    Tnew[j*(nx+2) + i] = T[j*(nx+2) + i];
                } else {
                    // Second-order central differencing for spatial derivatives
                    double d2Tdx2 = (T[j*(nx+2) + i+1] - 2.0*T[j*(nx+2) + i] + T[j*(nx+2) + i-1]) / (dx*dx);
                    double d2Tdy2 = (T[(j+1)*(nx+2) + i] - 2.0*T[j*(nx+2) + i] + T[(j-1)*(nx+2) + i]) / (dy*dy);
                    
                    // Explicit Euler time-stepping
                    Tnew[j*(nx+2) + i] = T[j*(nx+2) + i] + dt * (alpha * d2Tdx2 + beta * d2Tdy2);
                }
            }
        }
        
        // Update temperature field
        for (j = 1; j <= ny; j++) {
            for (i = 1; i <= nx; i++) {
                T[j*(nx+2) + i] = Tnew[j*(nx+2) + i];
            }
        }
        
        // Update time
        t += dt;
    }
    
    // Gather solution for comparison
    double *global_T = NULL;
    if (rank == 0) {
        global_T = (double*)malloc(nxglob * nyglob * sizeof(double));
    }
    
    // Create MPI datatype for local grid without ghost cells
    MPI_Datatype local_grid;
    MPI_Type_vector(ny, nx, nx+2, MPI_DOUBLE, &local_grid);
    MPI_Type_commit(&local_grid);
    
    // Gather local grids to rank 0
    if (rank == 0) {
        // Copy own data first
        for (j = 0; j < ny; j++) {
            for (i = 0; i < nx; i++) {
                global_T[j * nxglob + i] = T[(j+1)*(nx+2) + i+1];
            }
        }
        
        // Receive data from other processes
        for (int p = 1; p < size; p++) {
            int coords_p[2];
            MPI_Cart_coords(cart_comm, p, 2, coords_p);
            
            // Calculate starting indices for this process
            int start_i = 0;
            for (i = 0; i < coords_p[0]; i++) {
                start_i += nxglob / dims[0];
                if (i < nxglob % dims[0]) start_i++;
            }
            
            int start_j = 0;
            for (j = 0; j < coords_p[1]; j++) {
                start_j += nyglob / dims[1];
                if (j < nyglob % dims[1]) start_j++;
            }
            
            // Calculate dimensions for this process
            int nx_p = nxglob / dims[0];
            if (coords_p[0] < nxglob % dims[0]) nx_p++;
            
            int ny_p = nyglob / dims[1];
            if (coords_p[1] < nyglob % dims[1]) ny_p++;
            
            // Receive data
            double *temp_buf = (double*)malloc(nx_p * ny_p * sizeof(double));
            MPI_Recv(temp_buf, nx_p * ny_p, MPI_DOUBLE, p, 0, cart_comm, MPI_STATUS_IGNORE);
            
            // Copy data to global array
            for (j = 0; j < ny_p; j++) {
                for (i = 0; i < nx_p; i++) {
                    global_T[(start_j + j) * nxglob + (start_i + i)] = temp_buf[j * nx_p + i];
                }
            }
            
            free(temp_buf);
        }
    } else {
        // Send local grid to rank 0
        double *send_buf = (double*)malloc(nx * ny * sizeof(double));
        for (j = 0; j < ny; j++) {
            for (i = 0; i < nx; i++) {
                send_buf[j * nx + i] = T[(j+1)*(nx+2) + i+1];
            }
        }
        MPI_Send(send_buf, nx * ny, MPI_DOUBLE, 0, 0, cart_comm);
        free(send_buf);
    }
    
    // Compare with serial solution on rank 0
    if (rank == 0) {
        // Run serial solution
        double *T_serial = (double*)malloc(nxglob * nyglob * sizeof(double));
        double *Tnew_serial = (double*)malloc(nxglob * nyglob * sizeof(double));
        
        // Initialize serial solution
        for (j = 0; j < nyglob; j++) {
            for (i = 0; i < nxglob; i++) {
                if (i == 0 || i == nxglob - 1 || j == 0 || j == nyglob - 1) {
                    T_serial[j * nxglob + i] = 1.0;
                } else {
                    T_serial[j * nxglob + i] = 0.0;
                }
            }
        }
        
        // Run serial solution for 10 steps
        double t_serial = 0.0;
        for (istep = 0; istep < nsteps; istep++) {
            for (j = 1; j < nyglob - 1; j++) {
                for (i = 1; i < nxglob - 1; i++) {
                    double d2Tdx2 = (T_serial[j * nxglob + i+1] - 2.0*T_serial[j * nxglob + i] + T_serial[j * nxglob + i-1]) / (dx*dx);
                    double d2Tdy2 = (T_serial[(j+1) * nxglob + i] - 2.0*T_serial[j * nxglob + i] + T_serial[(j-1) * nxglob + i]) / (dy*dy);
                    
                    Tnew_serial[j * nxglob + i] = T_serial[j * nxglob + i] + dt * (alpha * d2Tdx2 + beta * d2Tdy2);
                }
            }
            
            // Update boundaries
            for (i = 0; i < nxglob; i++) {
                Tnew_serial[0 * nxglob + i] = T_serial[0 * nxglob + i];
                Tnew_serial[(nyglob-1) * nxglob + i] = T_serial[(nyglob-1) * nxglob + i];
            }
            
            for (j = 0; j < nyglob; j++) {
                Tnew_serial[j * nxglob + 0] = T_serial[j * nxglob + 0];
                Tnew_serial[j * nxglob + (nxglob-1)] = T_serial[j * nxglob + (nxglob-1)];
            }
            
            // Swap arrays
            double *temp = T_serial;
            T_serial = Tnew_serial;
            Tnew_serial = temp;
            
            t_serial += dt;
        }
        
        // Compare solutions
        double max_diff = 0.0;
        double l2_diff = 0.0;
        
        for (j = 0; j < nyglob; j++) {
            for (i = 0; i < nxglob; i++) {
                double diff = fabs(global_T[j * nxglob + i] - T_serial[j * nxglob + i]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
                l2_diff += diff * diff;
            }
        }
        
        l2_diff = sqrt(l2_diff / (nxglob * nyglob));
        
        printf("Comparison after %d time steps:\n", nsteps);
        printf("Max difference between serial and parallel: %e\n", max_diff);
        printf("L2 norm of difference: %e\n", l2_diff);
        
        // Write the differences to a file
        FILE *fdiff = fopen("difference.dat", "w");
        if (fdiff != NULL) {
            fprintf(fdiff, "Max difference: %e\n", max_diff);
            fprintf(fdiff, "L2 norm of difference: %e\n", l2_diff);
            fclose(fdiff);
        }
        
        // Clean up serial arrays
        free(T_serial);
        free(Tnew_serial);
    }
    
    // Clean up
    free(x);
    free(y);
    free(T);
    free(Tnew);
    free(xleftghost);
    free(xrightghost);
    free(ybotghost);
    free(ytopghost);
    MPI_Type_free(&local_grid);
    
    MPI_Finalize();
    return 0;
}


            