/*
 * Author: Sebastin Justeeson
 * Description: RBF network for one input and one output
 */




import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;

public class Rbfnetwork {
	private double hxSetArray[];
	private double nSetArray[];
	private double dSetArray[];
	private double xSetArray[];
	private double clusterCentersArray[];
	private double gaussianFunctionArray[];
	private double num;
	private double error = Double.MAX_VALUE;
	private final double maxAcceptableError = 0.05;
	private double variance[];
	private int i;
	private int elementIndex;
	private int j;
	private int epochs;
	private double obtainedOutput;
	private double hiddenLayerOutputArray[];
	private double obtainedOutputArray[];
	private double centerElement;
	private double learnRate;
	private int bases;
	private Random randomNum;
	private Set<Double> xSet;
	private Set<Double> nSet;
	private Set<Double> clusterCentersSet;
	private Iterator<Double> iterator;
	private List<Set<Double>> clusters;
	private Map<Double, Integer> classificationMap;
	private Map<Double, Double> pointClassificationCenterMap;
	private Node[] hiddenNodeArray;
	Node outputNode;
	
	public void init() {
		i = 0;
		j = 0;
		learnRate = 0.01;
		bases = 2;
		epochs = 100;
		
		hiddenLayerOutputArray = new double[bases];
		xSetArray = new double[75];
		hxSetArray = new double[75];
		nSetArray = new double[75];
		dSetArray = new double[75];
		obtainedOutputArray = new double[75];
		hiddenNodeArray = new Node[bases];
		gaussianFunctionArray = new double[75];
		clusterCentersArray = new double[bases];
		variance = new double[bases];
		outputNode = new Node();
		
		randomNum = new Random();
		xSet = new HashSet<Double>();
		nSet = new HashSet<Double>();
		clusterCentersSet = new HashSet<Double>();
		clusters = new ArrayList<Set<Double>>();
		classificationMap = new HashMap<Double, Integer>();
		pointClassificationCenterMap = new HashMap<Double, Double>();
		
		// Fill the list of clusters with sets
		for(i = 0; i < bases; i++) {
			clusters.add(new HashSet<Double>());
		}
		
		i = 0;
		
		// Generate x values
		while(xSet.size() < 75) {
			// Generate a random number in the interval [0 , 1]
			num = randomNum.nextDouble();
			
			if(!xSet.contains(num)){
				xSet.add(num);
				xSetArray[i] = num;
				i++;
			} 
		}
		
		// Obtain values of the function
		for(i = 0; i < 75; i++) {
			hxSetArray[i] = 0.5 + 0.4*Math.sin(2*3.14*xSetArray[i]);
		}
		
		i = 0;
		
		// Obtain noise values
		num = randomNum.nextDouble() * (1.0 - (-1.0)) + (-1.0);
		for(i = 0; i < 75; i++) {
			nSetArray[i] = num;
		}
		
		
		// Obtain the 75 data points
		for(i = 0; i < 75; i++) {
			dSetArray[i] = hxSetArray[i] + nSetArray[i];
		}	
		
		// Obtain the K cluster centers
		while(clusterCentersSet.size() < bases)
		{
			randomNum = new Random();
			elementIndex = randomNum.nextInt(75);
			centerElement = xSetArray[elementIndex];
			clusterCentersSet.add(centerElement);
		}
		
		// Label all the cluster centers with a number
		for(double d : clusterCentersSet) {
			classificationMap.put(d, j);
			j++;
		}
		
		// Transfer all the cluster centers to an array
		for(double d : classificationMap.keySet()) {
			clusterCentersArray[classificationMap.get(d)] = d;
		}
		
	}
	
	public void GaussianFunction() {
		double center;
		int centerIndex;
		double var;
		double result;
		
		// Calculate the gaussian function associated with every data point
		for(i = 0; i < 75; i++) {
			center = pointClassificationCenterMap.get(xSetArray[i]);
			centerIndex = classificationMap.get(center);
			var = variance[centerIndex];
			result = Math.pow(Math.abs(xSetArray[i] - pointClassificationCenterMap.get(xSetArray[i])), 2);
			result = result / (2 * var);
			result = Math.exp(-result);	
			gaussianFunctionArray[i] = result;
		}
	}
	
	public void KMeansClustering() {
		double newCenter;
		double oldCenter;
		int updatedCenters;
		
		// Repeat the algorithm until all centers have converged
		do {
			updatedCenters = 0;
			for(Set<Double> s : clusters) {
				s.clear();
			}

		// Classify all the data elements
		for(Double d : xSetArray) {
			FindNearestNeighborAndClassify(d);
		}
		
		
		// Calculate new centers
		for(i = 0; i < bases; i++) {
			oldCenter = clusterCentersArray[i];
			newCenter = CalculateNewCenter(clusters.get(i));
			// Update the center value in the map and array
			if(oldCenter != newCenter) {
				classificationMap.put(newCenter, i);
				classificationMap.remove(oldCenter);
				clusterCentersArray[i] = newCenter;
				clusterCentersSet.remove(oldCenter);
				clusterCentersSet.add(newCenter);
				updatedCenters++;
			}
		}
		} while(updatedCenters != 0);
		
		FindClusterVariance();	
	}
	
	public double CalculateNewCenter(Set<Double> cluster) {
		double total = 0;
		double setSize = cluster.size();
		double avg = 0;
		
		for(Double d : cluster) {
			total = total + d;
		}
		
		avg = total / setSize;
		
		return avg;
	}
	
	public void FindNearestNeighborAndClassify(double elem) {
		double smallestDistance = Double.MAX_VALUE;
		double distance;
		double closestCenter = 0;
		int clusterIndex;
		
		// Find cluster center nearest to given point
		for(Double cen : clusterCentersSet) {
			distance = Math.pow(Math.abs(elem - cen), 2);
			if(distance < smallestDistance) {
				smallestDistance = distance;
				closestCenter = cen;
			}
		}
		
		// Find the correct cluster to classify the element under
		clusterIndex = classificationMap.get(closestCenter);
		clusters.get(clusterIndex).add(elem);
		
		// Map the closest center to every point
		if(pointClassificationCenterMap.containsKey(elem)) {
			pointClassificationCenterMap.remove(elem);
			pointClassificationCenterMap.put(elem, closestCenter);
		}
		else {
			pointClassificationCenterMap.put(elem, closestCenter);
		}
	}
	
	public void FindClusterVariance() {
		double cen;
		Set<Double> tempSet;
		double total = 0;
		double size = 0;
		double meanVariance = 0;
	    boolean singleElemCheck = false;
	    int singleElemIndex = 0;
		
		// Calculate the variance for every cluster
		for(i = 0; i < bases; i++) {
			cen = clusterCentersArray[i];
			tempSet = clusters.get(i);
			size = tempSet.size();
			
			if(size == 1) {
				singleElemIndex = i;
				singleElemCheck = true;
			}
			else {
				for(Double d : tempSet) {
					total += Math.pow(Math.abs(d - cen), 2);
				}
				
				variance[i] = total / size;
				meanVariance += variance[i];
			}			
		}
		
		// If only one element, then assign mean variance of other clusters to it
		if(singleElemCheck) {
			 variance[singleElemIndex] = meanVariance / (bases - 1);
		}		
	}
	
	public void LMSFunction() {
		for(i = 0; i < bases; i++) {
			hiddenNodeArray[i] = new Node();
		}
		
		// Repeat for every set of input
		for(j = 0; j < epochs; j++) {
			for(i = 0; i < 75; i++) {
				while(Math.abs(error) > maxAcceptableError) {
					double hiddenNodeSum = 0;
					// Obtain outputs for hidden layer nodes
					for(int k = 0; k < bases; k++) {
						hiddenLayerOutputArray[k] = (hiddenNodeArray[k].weight * gaussianFunctionArray[i]);
						hiddenNodeSum += hiddenLayerOutputArray[k];
					}
					obtainedOutputArray[i] = hiddenNodeSum + outputNode.bias;
					error = (dSetArray[i] - obtainedOutputArray[i]);
					
					if(Math.abs(error) < maxAcceptableError) {
						error = Double.MAX_VALUE;
						break;
					}
					
					// Calculate the new weights for all nodes
					for(int k = 0; k < bases; k++) {
						hiddenNodeArray[k].weight = hiddenNodeArray[k].weight + (learnRate * error * xSetArray[i]);
					}
				}
			}
		}
		
		
		
		// Print the results
		System.out.println("Data point         Function output   Obtained output");
		System.out.println("------------------------------------------------------");
		for(i = 0; i < 75; i++) {
			System.out.println(xSetArray[i] + " " + hxSetArray[i] + " " + obtainedOutputArray[i]);		
		}
	}
	
	
	public static void main(String args[]) {
		Rbfnetwork rb = new Rbfnetwork();
		rb.init();
		rb.KMeansClustering();
		rb.GaussianFunction();
		rb.LMSFunction();
	}
}