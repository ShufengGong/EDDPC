package eddpc.dpclu;

/**
 * @author gongsf
 * 
 */

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
//import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.*;

import eddpc.util.MyConf;
import eddpc.util.Point;

public class FirstJob {

	public static class MapperPoint extends
			Mapper<Object, Text, IntWritable, Text> {

		public Point point;
		private int dim;
		private int partionNum;
		private Point pivots[];
		private float psdistance[][];
		private int dc;

		@Override
		public void setup(Context context) throws IOException {
			Configuration conf = context.getConfiguration();
			dim = conf.getInt("dp.data.dim", 57);
			partionNum = conf.getInt("dp.pivotNum", 200);
			dc = conf.getInt("dp.offset", 200);
			pivots = new Point[partionNum];
			Path cachePath[] = DistributedCache.getLocalCacheFiles(conf);
			if (cachePath != null && cachePath.length > 0) {
				BufferedReader br = new BufferedReader(new FileReader(
						cachePath[0].toString()));
				String line;
				while ((line = br.readLine()) != null) {
					int index = 0;
					int partId = 0;
					float data[] = new float[dim];
					StringTokenizer st = new StringTokenizer(line);
					if (st.hasMoreTokens()) {
						partId = Integer.parseInt(st.nextToken());
						index = Integer.parseInt(st.nextToken());
					}
					for (int j = 0; st.hasMoreTokens(); j++) {
						data[j] = Float.parseFloat(st.nextToken());
					}

					pivots[partId] = new Point(index, data);
				}
				br.close();
				psdistance = new float[partionNum][partionNum];
				BufferedReader br2 = new BufferedReader(new FileReader(
						cachePath[1].toString()));
				String line2;
				while ((line2 = br2.readLine()) != null) {
					StringTokenizer st2 = new StringTokenizer(line2, ",");
					int i = Integer.parseInt(st2.nextToken());
					while (st2.hasMoreTokens()) {
						StringTokenizer st3 = new StringTokenizer(
								st2.nextToken());
						int j = Integer.parseInt(st3.nextToken());
						float distance = Float.parseFloat(st3.nextToken());
						psdistance[i][j] = distance;
					}
				}
				br2.close();
			}
		}

		public void map(Object key, Text value, Context context)
				throws IOException, InterruptedException {
			String line = value.toString();
			StringTokenizer stk = new StringTokenizer(line);
			int pid = Integer.parseInt(stk.nextToken());
			float data[] = new float[dim];
			for (int i = 0; stk.hasMoreTokens(); i++) {
				data[i] = Float.parseFloat(stk.nextToken());
			}

			point = new Point(pid, data);
			float minDistance = Float.MAX_VALUE;
			int minIndex = 0;
			float toPDistence[] = new float[partionNum];
			for (int i = 0; i < partionNum; i++) {
				float distence = point.getDistence(pivots[i]);
				toPDistence[i] = distence;
				if (distence < minDistance) {
					minDistance = distence;
					minIndex = i;
				}
			}

			context.write(new IntWritable(minIndex), new Text("o: "
					+ minDistance + " " + line));
			for (int i = 0; i < partionNum; i++) {
				float disToHy = 0;
				float disToi = toPDistence[i];
				if (minIndex < i) {
					disToHy = (disToi * disToi - minDistance * minDistance)
							/ (2 * psdistance[minIndex][i]);
					if (disToHy < dc)
						context.write(new IntWritable(i), new Text("e: "
								+ line));
				}
				if (minIndex > i) {
					disToHy = (disToi * disToi - minDistance * minDistance)
							/ (2 * psdistance[i][minIndex]);
					if (disToHy < dc)
						context.write(new IntWritable(i), new Text("e: "
								+ line));
				}
			}

		}

	}

	public static class ReducerPoint extends
			Reducer<IntWritable, Text, IntWritable, Text> {

		private int dim;
		private int dc;
		private MultipleOutputs mos;

		protected void setup(Context context) {
			Configuration conf = context.getConfiguration();
			dim = conf.getInt("dp.data.dim", 57);
			dc = conf.getInt("dp.offset", 200);
			mos = new MultipleOutputs(context);
		}

		public void reduce(IntWritable key, Iterable<Text> values,
				Context context) throws IOException, InterruptedException {
			float maxLocalDelt = 0;
			int maxRouId = 0;
			int maxLocalRou = 0;
			int minLocalRou = Integer.MAX_VALUE;
			ArrayList<Point> localPointList = new ArrayList<Point>();
			ArrayList<Point> extendPointList = new ArrayList<Point>();
			ArrayList<Float> distanceToPList = new ArrayList<Float>();
			float localMaxBound = 0;
			for (Text value : values) {
				String s = value.toString();
				if (s.startsWith("o")) {
					StringTokenizer str = new StringTokenizer(s);
					str.nextToken();// skip o:
					float distance = Float.parseFloat(str.nextToken());
					if (localMaxBound < distance){
						localMaxBound = distance;
					}
					distanceToPList.add(distance);
					int index = Integer.parseInt(str.nextToken());
					float data[] = new float[dim];
					for (int i = 0; str.hasMoreTokens(); i++){
						data[i] = Float.parseFloat(str.nextToken());
					}

					Point p = new Point(index, data);
					localPointList.add(p);
				} else {
					StringTokenizer str = new StringTokenizer(s);
					str.nextToken();
					int index = Integer.parseInt(str.nextToken());
					float data[] = new float[dim];
					for (int i = 0; str.hasMoreTokens(); i++){
						data[i] = Float.parseFloat(str.nextToken());
					}

					Point p = new Point(index, data);
					extendPointList.add(p);
				}
			}

//			System.out.println(distanceToPList.size());
			
			int bucketPointNum = localPointList.size();
//			System.out.println("partId:" + key.get() +";numbero:" + bucketPointNum);
//			System.out.println("partId:" + key.get() +";numbere:" + extendPointList.size());
			if(bucketPointNum == 0){
				System.out.println("no points");
				if(bucketPointNum != distanceToPList.size()){
					System.out.println("length is not equal");
				}
				return;
			}
			// float distence[][] = new float[bucketPointNum][bucketPointNum];
			float dis = 0;
			for (int i = 0; i < bucketPointNum; i++) {
				// distence[i][i] = 0.0F;
				Point p = localPointList.get(i);
				p.rou = 1;
				for (int j = 0; j < i; j++) {
					dis = p.getDistence(localPointList.get(j));
					if (dis < dc) {
						p.rou++;
						localPointList.get(j).rou++;
					}
				}

				for (int k = 0; k < extendPointList.size(); k++) {
					if (p.getDistence(extendPointList.get(k)) < dc) {
						p.rou++;
					}
				}
			}


			for (int i = 0; i < bucketPointNum; i++) {
				Point p = localPointList.get(i);
				p.delt = Float.MAX_VALUE;
				for (int j = 0; j < i; j++) {
					Point pj = localPointList.get(j);
					dis = p.getDistence(pj);
					if (p.rou < pj.rou && p.delt > dis) {
						p.delt = dis;
						p.dep = localPointList.get(j).index;
					}
					if (p.rou > pj.rou && pj.delt > dis) {
						pj.delt = dis;
						pj.dep = localPointList.get(i).index;
					}
				}

				if (maxLocalRou < p.rou) {
					maxLocalRou = p.rou;
					maxRouId = i;
				}
				if (minLocalRou > p.rou) {
					minLocalRou = p.rou;
				}
			}

			for (int i = 0; i < bucketPointNum; i++) {
				Point p = localPointList.get(i);
				if (maxLocalDelt < p.delt && p.delt != Float.MAX_VALUE)
					maxLocalDelt = p.delt;
				StringBuilder sb = new StringBuilder(" ");
//                sb.append(distanceToPList.get(i));
                sb.append(" ").append(p.rou);
                sb.append(" ").append(p.delt);
                sb.append(" ").append(p.dep);
                sb.append(" ").append(p.toString());
                context.write(key, new Text(sb.toString()));
//				context.write(key, new Text(sb.toString()));
			}

			mos.write("maxBound", key, new Text(localMaxBound + ""));
			mos.write("maxDelt", key, new Text(maxLocalDelt + ""));
			mos.write("minRou", key, new Text(minLocalRou + ""));
			mos.write("maxRou", key, new Text(maxLocalRou + ""));
			mos.write("maxRDis", key, new Text(distanceToPList.get(maxRouId)
					+ ""));
		}

		protected void cleanup(Context context) throws IOException,
				InterruptedException {
			mos.close();
		}
	}

	public static void mergeFile(String inpath, String kind, String outpath,
			FileSystem fs) {
		try {
			OutputStream out = fs.create(new Path(outpath), true);
			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(out));
			String infile = null;
			Path infilePath = null;
			String s = null;
			FileStatus afilestatus[];
			int j = (afilestatus = fs.listStatus(new Path(inpath))).length;
			for (int i = 0; i < j; i++) {
				FileStatus file = afilestatus[i];
				infilePath = file.getPath();
				infile = infilePath.getName();
				if (infile.startsWith(kind)) {
					InputStream ins = fs.open(infilePath);
					BufferedReader br = new BufferedReader(
							new InputStreamReader(ins));
					while ((s = br.readLine()) != null)
						bw.write((new StringBuilder(String.valueOf(s))).append(
								"\n").toString());
					br.close();
					ins.close();
					fs.delete(infilePath, true);
				}
			}

			bw.close();
			out.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void getRho(String args[]) {
		Configuration conf = new Configuration();
		String input = null;
		String output = null;
		int dim = 0;
		int sampleNum = 0;
		int dc = 0;
		int rnum = 0;
		for (int i = 0; i < args.length; i++) {
			if (args[i].equals("-dim"))
				dim = Integer.parseInt(args[++i]);
			if (args[i].equals("-sampleNum"))
				sampleNum = Integer.parseInt(args[++i]);
			if (args[i].equals("-dc"))
				dc = Integer.parseInt(args[++i]);
			if (args[i].equals("-in"))
				input = args[++i];
			if (args[i].equals("-out"))
				output = args[++i];
			if (args[i].equals("-rnum"))
				rnum = Integer.parseInt(args[++i]);
		}

		if (dim == 0 || sampleNum == 0 || dc == 0 || rnum == 0) {
			System.err
					.println("invalid parameters =======================================");
			return;
		}
		if (input == null || output == null) {
			System.err
					.println("invalid parameters =======================================");
			return;
		}
		conf.setInt("dp.data.dim", dim);
		conf.setInt("dp.pivotNum", sampleNum);
		try {
			DistributedCache.addCacheFile(new URI(MyConf.pivots), conf);
			DistributedCache.addCacheFile(new URI(MyConf.psdistance), conf);
			FileSystem fs = FileSystem.get(conf);
			Job job = new Job(conf, "firstJob");
			job.setJarByClass(FirstJob.class);
			job.setMapperClass(MapperPoint.class);
			// job.setInputFormatClass(org/apache/hadoop/mapreduce/lib/input/KeyValueTextInputFormat);
			job.setMapOutputKeyClass(IntWritable.class);
			job.setMapOutputValueClass(Text.class);
			job.setReducerClass(ReducerPoint.class);
			job.setNumReduceTasks(rnum);
			job.setOutputKeyClass(IntWritable.class);
			job.setOutputValueClass(Text.class);
			MultipleOutputs.addNamedOutput(job, "maxBound",
					TextOutputFormat.class, IntWritable.class, Text.class);
			MultipleOutputs.addNamedOutput(job, "maxDelt",
					TextOutputFormat.class, IntWritable.class, Text.class);
			MultipleOutputs.addNamedOutput(job, "minRou",
					TextOutputFormat.class, IntWritable.class, Text.class);
			MultipleOutputs.addNamedOutput(job, "maxRou",
					TextOutputFormat.class, IntWritable.class, Text.class);
			MultipleOutputs.addNamedOutput(job, "maxRDis",
					TextOutputFormat.class, IntWritable.class, Text.class);
			FileInputFormat.addInputPath(job, new Path(input));
			FileOutputFormat.setOutputPath(job, new Path(output));
			job.waitForCompletion(true);
			System.out.println("start merge =================================");
			mergeFile(output, "maxBound", MyConf.maxBound, fs);
			mergeFile(output, "maxDelt", MyConf.maxDelt, fs);
			mergeFile(output, "maxRou", MyConf.maxRou, fs);
			mergeFile(output, "minRou", MyConf.minRou, fs);
			mergeFile(output, "maxRDis", MyConf.maxRouDis, fs);
			System.out.println("merged =================================");
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (URISyntaxException e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args){
		getRho(args);
	}

}
