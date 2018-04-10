package eddpc.util;

/**
 * @author gongsf
 *
 */

public class Point {

	public int index;
	public float data[];
	public int dim;
	public int rou;
	public float delt;
	public int dep;

	public Point() {
	}

	public Point(Point p) {
		index = p.index;
		data = p.data;
		dim = data.length;
	}

	public Point(int dim) {
		data = new float[dim];
		this.dim = dim;
	}

	public Point(int index, float data[]) {
		this.index = index;
		this.data = data;
		dim = data.length;
	}

	public float getDistence(Point p) {
		double sum = 0.0D;
		for (int i = 0; i < data.length; i++)
			sum += (data[i] - p.data[i]) * (data[i] - p.data[i]);

		return (float) Math.sqrt(sum);
	}

	public int getDim() {
		return data.length;
	}

	public String toString() {
		StringBuilder sb = new StringBuilder(" ");
		sb.append(index);
		for (int i = 0; i < dim; i++)
			sb.append(" "+ data[i]);

		return sb.toString();
	}

}
