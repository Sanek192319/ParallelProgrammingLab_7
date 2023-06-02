import mpi.MPI;

public class Main {
    private static final int rowsA = 4;
    private static final int colsA = 4;
    private static final int colsB = 4;
    private static final int MASTER = 0;
    public static void main(String[] args) {
        double[][] a = new double[rowsA][colsA];
        double[][] b = new double[colsA][colsB];
        double[][] c = new double[rowsA][colsB];
        MPI.Init(args);
        int numTasks = MPI.COMM_WORLD.Size();
        int rank = MPI.COMM_WORLD.Rank();

        if (numTasks < 2) {
            System.out.println("Need at least two MPI tasks. Quitting...\n");
            MPI.COMM_WORLD.Abort(1);
        }
        long start = 0;
        if(rank == MASTER) {
            a = InputMautix(rowsA,colsA,1);

            b = InputMautix(colsA,colsB,1);
            start = System.currentTimeMillis();
        }
        int rowsPerThread = rowsA / numTasks;
        int extra = rowsA % numTasks;

        int[] rowsCounts = new int[numTasks];
        int[] offsetCounts = new int[numTasks];
        for(int i = 0; i < numTasks; i++) {
            rowsCounts[i] = i < extra ? rowsPerThread + 1 : rowsPerThread;
            offsetCounts[i] = i == MASTER ? 0 : offsetCounts[i-1] + rowsCounts[i-1];
        }

        int rowsForThisTask = rowsCounts[rank];
        double[][] aRows = new double[rowsForThisTask][colsA];
        MPI.COMM_WORLD.Scatterv(
                a, 0, rowsCounts, offsetCounts,MPI.OBJECT,
                aRows, 0, rowsForThisTask, MPI.OBJECT,
                MASTER
        );
        MPI.COMM_WORLD.Bcast(b, 0, colsA, MPI.OBJECT, MASTER);

        double[][] cRows = new double[rowsForThisTask][colsB];
        for (int k = 0; k < colsB; k++) {
            for (int i = 0; i < rowsForThisTask; i++) {
                for (int j = 0; j < colsA; j++) {
                    cRows[i][k] += aRows[i][j] * b[j][k];
                }
            }
        }

        MPI.COMM_WORLD.Allgatherv(
                cRows, 0, rowsForThisTask, MPI.OBJECT,
                c, 0, rowsCounts, offsetCounts, MPI.OBJECT
        );

        if (rank == MASTER) {
            var endTime = System.currentTimeMillis();
            var dur = endTime - start;

            for(int i = 0; i < rowsA; i++) {
                for (int j = 0; j < colsB; j++) {
                    System.out.print(c[i][j] +" ");
                }
                System.out.print('\n');
            }
            System.out.println("End with time: " + dur + " ms");
        }
        MPI.Finalize();
    }
    public static double[][] InputMautix(int rows,int cols,double value)
    {
        double[][] result = new double[rows][cols];
        for(int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = value;
            }
        }
        return result;
    }
}