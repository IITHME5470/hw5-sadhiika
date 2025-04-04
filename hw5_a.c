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
    double t, time_taken;
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
    
    // Ghost arrays for halo exchange testing
    xleftghost = (double*)malloc(ny * sizeof(double));
    xrightghost = (double*)malloc(ny * sizeof(double));
    ybotghost = (double*)malloc(nx * sizeof(double));
    ytopghost = (double*)malloc(nx * sizeof(double));
    
    // Generate grid
    grid(nx, nxglob, istglob, ienglob, xst, xen, &x[1], &dx);
    grid(ny, nyglob, jstglob, jenglob, yst, yen, &y[1], &dy);
    
    // Initialize temperature field with rank*10+i+j for halo exchange testing
    for (j = 1; j <= ny; j++) {
        for (i = 1; i <= nx; i++) {
            T[j*(nx+2) + i] = rank*10 + i + j;
        }
    }
    
    // Perform halo exchange
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
    
    // Write ghost values to file for verification
    sprintf(fname, "ghost_rank%d.dat", rank);
    fout = fopen(fname, "w");
    fprintf(fout, "xleftghost: ");
    for (j = 0; j < ny; j++) {
        fprintf(fout, "%lf ", T[(j+1)*(nx+2) + 0]);
    }
    fprintf(fout, "\nxrightghost: ");
    for (j = 0; j < ny; j++) {
        fprintf(fout, "%lf ", T[(j+1)*(nx+2) + nx+1]);
    }
    fprintf(fout, "\nybotghost: ");
    for (i = 0; i < nx; i++) {
        fprintf(fout, "%lf ", T[0*(nx+2) + i+1]);
    }
    fprintf(fout, "\nytopghost: ");
    for (i = 0; i < nx; i++) {
        fprintf(fout, "%lf ", T[(ny+1)*(nx+2) + i+1]);
    }
    fprintf(fout, "\n");
    fclose(fout);
    
    // Now run the actual simulation
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
    nsteps = (int)(tend / dt);
    
    // Start timing
    start_time = MPI_Wtime();
    
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
        
        // Output at selected time steps
        if (istep % 100 == 0 || istep == nsteps - 1) {
            // Gather data to rank 0 for output
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
                        global_T[j * nxglob + i] = T[j*(nx+2) + i + 1];
                    }
                }
                
                // Receive data from other processes
                for (int p = 1; p < size; p++) {
                    int coords_p[2];
                    MPI_Cart_coords(cart_comm, p, 2, coords_p);
                    int nx_p = (coords_p[0] < nxglob % dims[0]) ? nx + 1 : nx;
                    int ny_p = (coords_p[1] < nyglob % dims[1]) ? ny + 1 : ny;
                    int start_i = coords_p[0] * nx;
                    int start_j = coords_p[1] * ny;
                    
                    MPI_Recv(&global_T[start_j * nxglob + start_i], 1, local_grid, p, 0, cart_comm, MPI_STATUS_IGNORE);
                }
            } else {
                // Send local grid to rank 0
                MPI_Send(&T[1*(nx+2) + 1], 1, local_grid, 0, 0, cart_comm);
            }
            
            // Write global temperature field to file
            if (rank == 0) {
                char filename[256];
                sprintf(filename, "T_x_y_%06d.dat", istep);
                FILE *fp = fopen(filename, "w");
                if (fp == NULL) {
                    printf("Cannot open file %s\n", filename);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                
                for (j = 0; j < nyglob; j++) {
                    for (i = 0; i < nxglob; i++) {
                        fprintf(fp, "%f %f %f\n", xstglob + i*dx, ystglob + j*dy, global_T[j*nxglob + i]);
                    }
                }
                fclose(fp);
            }
                    }
                }
                
                // End timing
                end_time = MPI_Wtime();
                
                if (rank == 0) {
                    printf("Computation completed in %d time steps\n", nsteps);
                    printf("Total time: %f seconds\n", end_time - start_time);
                    printf("Time per step: %f seconds\n", (end_time - start_time) / nsteps);
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
                if (rank == 0) {
                    free(global_T);
                }
                MPI_Type_free(&local_grid);
                
                MPI_Finalize();
                return 0;
            }
            
                    
            
