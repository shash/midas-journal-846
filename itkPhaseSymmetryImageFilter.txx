
#ifndef __itkPhaseSymmetryImageFilter_txx
#define __itkPhaseSymmetryImageFilter_txx
#include "itkPhaseSymmetryImageFilter.h"
#include <string>
#include <sstream>

namespace itk
{

	template <class TInputImage, class TOutputImage>
	PhaseSymmetryImageFilter<TInputImage,TOutputImage>::PhaseSymmetryImageFilter()
	{
		m_MultiplyImageFilter = MultiplyImageFilterType::New();

		m_FFTFilter  = FFTFilterType::New();
		m_IFFTFilter = IFFTFilterType::New();

		//Create 2 initialze wavelengths
		m_Wavelengths.SetSize(2,TInputImage::ImageDimension);
		for( int i=0; i < TInputImage::ImageDimension; i++)
		{
			m_Wavelengths(0,i) = 10.0;
			m_Wavelengths(1,i) = 20.0;
	
		}

		//Set basic orientations
		m_Orientations.SetSize(TInputImage::ImageDimension,TInputImage::ImageDimension);
		for( int i=0; i < TInputImage::ImageDimension; i++)
		{
			for( int j=0; j < TInputImage::ImageDimension; j++)
			{
				if(i==j)
				{
					m_Orientations(i,j) = 1.0;
				}
				else
				{
					m_Orientations(i,j) = 0.0;
				}
				
			}
	
		}

		//Defaults
		m_AngleBandwidth=3.14159265;
		m_Sigma=0.55;
		m_T=10.0;
		m_Polarity=0;


	}

	template <class TInputImage, class TOutputImage>
	void PhaseSymmetryImageFilter<TInputImage,TOutputImage>::Initialize( void )
	{
		typename TInputImage::SizeType  inputSize;
		typename TInputImage::IndexType inputIndex;
		typename Superclass::OutputImagePointer output = this->GetOutput();
		typename Superclass::InputImagePointer input = const_cast< TInputImage *>( this->GetInput() );

		inputIndex = input->GetLargestPossibleRegion().GetIndex();
		inputSize = input->GetLargestPossibleRegion().GetSize();
		int ndims = int(TInputImage::ImageDimension);

		typename LogGaborFreqImageSourceType::Pointer LogGaborKernel =  LogGaborFreqImageSourceType::New();
		typename SteerableFiltersFreqImageSourceType::Pointer SteerableFilterKernel =  SteerableFiltersFreqImageSourceType::New();
		typename ButterworthKernelFreqImageSourceType::Pointer ButterworthFilterKernel =  ButterworthKernelFreqImageSourceType::New();

		//Initialize log gabor kernels
		LogGaborKernel->SetOrigin(this->GetInput()->GetOrigin());
		LogGaborKernel->SetSpacing(this->GetInput()->GetSpacing());
		LogGaborKernel->SetSize(inputSize);
		LogGaborKernel->ReleaseDataFlagOn();

		//Initialize directionality kernels
		SteerableFilterKernel->SetOrigin(this->GetInput()->GetOrigin());
		SteerableFilterKernel->SetSpacing(this->GetInput()->GetSpacing());
		SteerableFilterKernel->SetSize(inputSize);
		SteerableFilterKernel->ReleaseDataFlagOn();

		//Initialize low pass filter kernel
		ButterworthFilterKernel->SetOrigin(this->GetInput()->GetOrigin());
		ButterworthFilterKernel->SetSpacing(this->GetInput()->GetSpacing());
		ButterworthFilterKernel->SetSize(inputSize);
		ButterworthFilterKernel->SetCutoff(0.4);
		ButterworthFilterKernel->SetOrder(10.0);
		SteerableFilterKernel->ReleaseDataFlagOn();

		ArrayType wv;
		ArrayType sig;
		ArrayType orientation;


		FloatImageStack tempStack;
		FloatImageStack lgStack;
		FloatImageStack sfStack;

		//Create filter bank by multiplying log gabor filters with directional filters
		DoubleFFTShiftImageFilterType::Pointer FFTShiftFilter = DoubleFFTShiftImageFilterType::New();

		for( unsigned int w=0; w < m_Wavelengths.rows(); w++)
		{
			for(int i=0; i < ndims;i++)
			{	
				wv[i]=m_Wavelengths.get(w,i);
			}
			LogGaborKernel->SetWavelengths(wv);
			LogGaborKernel->SetSigma(m_Sigma);
			m_MultiplyImageFilter->SetInput1(LogGaborKernel->GetOutput());
			m_MultiplyImageFilter->SetInput2(ButterworthFilterKernel->GetOutput());
			m_MultiplyImageFilter->Update();
			FloatImageType::Pointer gabor = m_MultiplyImageFilter->GetOutput();
			gabor->DisconnectPipeline();
			tempStack.clear();
			for( int o=0; o<m_Orientations.rows(); o++)
			{
				for( int d=0; d<ndims; d++)
				{
					orientation[d] = m_Orientations.get(o,d);

				}
				SteerableFilterKernel->SetOrientation(orientation);
				SteerableFilterKernel->SetAngularBandwidth(m_AngleBandwidth);
				SteerableFilterKernel->Update();
				FloatImageType::Pointer steerable = SteerableFilterKernel->GetOutput();
				m_MultiplyImageFilter->SetInput1(gabor);
				m_MultiplyImageFilter->SetInput2(steerable);
				FFTShiftFilter->SetInput(m_MultiplyImageFilter->GetOutput());
				FFTShiftFilter->Update();
				tempStack.push_back(FFTShiftFilter->GetOutput());
				tempStack[o]->DisconnectPipeline();
			}
			m_FilterBank.push_back(tempStack);
		}
		m_MultiplyImageFilter = MultiplyImageFilterType::New();
	}


	template <class TInputImage, class TOutputImage>
	void PhaseSymmetryImageFilter<TInputImage,TOutputImage>::GenerateData( void )
	{
		
		//typedef itk::ImageFileWriter< FloatImageType> WriterType;
		//WriterType::Pointer writer= WriterType::New();
		
		typename TInputImage::SizeType  inputSize;
		typename TInputImage::IndexType inputIndex;
		typename Superclass::OutputImagePointer output = this->GetOutput();
		typename Superclass::InputImagePointer input = const_cast< TInputImage *>( this->GetInput() );

		inputIndex = input->GetLargestPossibleRegion().GetIndex();
		inputSize = input->GetLargestPossibleRegion().GetSize();
		int ndims = int(TInputImage::ImageDimension);

		double initialPoint=0;
		double epsilon = 0.0001;

		ComplexImageType::Pointer finput;
		ComplexImageType::Pointer bpinput;

		m_FFTFilter->SetInput(input);
		m_FFTFilter->Update();
		finput = m_FFTFilter->GetOutput();
		finput->DisconnectPipeline();


		//Get the pixel count.  We need to divide the IFFT output by this because using the inverse FFT for
		//complex to complex doesn't seem to work.   So instead, we use the forward transform and divide by pixelNum
		double pxlCount = 1.0;
		for(int i=0; i < ndims; i++)
		{
			pxlCount = pxlCount*double(inputSize[i]);
		}


		typename FloatImageType::Pointer totalAmplitude;
		typename FloatImageType::Pointer totalEnergy = FloatImageType::New();

		//Matlab style initalization, because these images accumulate over each loop
		//Therefore, they initially all zeros
		typename TInputImage::RegionType inputRegion = input->GetLargestPossibleRegion();

		output->CopyInformation(input);
		output->SetRegions(inputRegion);
		output->Allocate();
		output->FillBuffer(0);

		totalAmplitude = output;

		totalEnergy->CopyInformation(input);
		totalEnergy->SetRegions(inputRegion);
		totalEnergy->Allocate();
		totalEnergy->FillBuffer(0);
		
		ComplexImageType::Pointer multiplied = ComplexImageType::New();
		multiplied->CopyInformation(finput);
		multiplied->SetRegions(inputRegion);
		multiplied->Allocate();
		multiplied->FillBuffer(0);

		m_IFFTFilter->ReleaseDataFlagOn();

		for( int o=0; o < m_Orientations.rows(); o++)
		{
			for( int w=0; w < m_Wavelengths.rows(); w++)
			{
				//Multiply filters by the input image in fourier domain
				ComplexImageIteratorType inputIterator(finput, inputRegion);
				FloatImageIteratorType filterBankIterator(m_FilterBank[w][o], inputRegion);
				ComplexImageIteratorType multipliedImageIterator(multiplied, inputRegion);
				while(!inputIterator.IsAtEnd())
				{
					std::complex<float> filtered = inputIterator.Value() * filterBankIterator.Value();
					filtered /= pxlCount;

					multipliedImageIterator.Value() = filtered;
					++inputIterator;
					++filterBankIterator;
					++multipliedImageIterator;
				}

				m_IFFTFilter->SetInput(multiplied);
				m_IFFTFilter->Update();
				bpinput=m_IFFTFilter->GetOutput();
				//Get mag, real and imag of the band passed images

				FloatImageIteratorType amplitudeImageIterator(totalAmplitude, inputRegion);
				ComplexImageIteratorType bpInputIterator(bpinput, bpinput->GetLargestPossibleRegion());
				while(!bpInputIterator.IsAtEnd())
				{
					float real = bpInputIterator.Value().real();
					float imag = bpInputIterator.Value().imag();
					amplitudeImageIterator.Value() += sqrt(real * real + imag * imag);
					++amplitudeImageIterator;
					++bpInputIterator;
				}
	
				bpInputIterator.GoToBegin();

				typename FloatImageType::RegionType region = totalEnergy->GetLargestPossibleRegion();

				FloatImageIteratorType energyIterator(totalEnergy, region);
				
				//Use appropraite equation depending on polarity
				if(m_Polarity==0)
				{
					while (!bpInputIterator.IsAtEnd())
					{
						float absReal = fabs(bpInputIterator.Value().real());
						float absImag = fabs(bpInputIterator.Value().imag());
						float energy = absReal - absImag;
						energyIterator.Value() += energy - m_T;
						++bpInputIterator;
						++energyIterator;
					}
				}
				else if(m_Polarity==1)
				{
					while (!bpInputIterator.IsAtEnd())
					{
						float real = bpInputIterator.Value().real();
						float absImag = fabs(bpInputIterator.Value().imag());
						float energy = real - absImag;
						energyIterator.Value() += energy - m_T;
						++bpInputIterator;
						++energyIterator;
					}				
				}
				else if(m_Polarity==-1)
				{
					while (!bpInputIterator.IsAtEnd())
					{
						float real = bpInputIterator.Value().real();
						float absImag = fabs(bpInputIterator.Value().imag());
						float energy = -real - absImag;
						++bpInputIterator;
						++energyIterator;
					}
				}
			}
		}

		FloatImageIteratorType outputIterator(output, inputRegion);
		FloatImageIteratorType energyIterator(totalEnergy, inputRegion);

		while (!outputIterator.IsAtEnd())
		{
			outputIterator.Value() = std::max(0.f, energyIterator.Value()) / outputIterator.Value();
			++outputIterator;
			++energyIterator;
		}
		itkDebugMacro("GenerateOutputInformation End");
	}


	template <class TInputImage, class TOutputImage>
	void PhaseSymmetryImageFilter<TInputImage,TOutputImage>::GenerateInputRequestedRegion()
	{
		itkDebugMacro("GenerateInputRequestedRegion Start");
		Superclass::GenerateInputRequestedRegion();

		if ( this->GetInput() )
		{
			typename TInputImage::RegionType RequestedRegion;
			typename TInputImage::SizeType  inputSize;
			typename TInputImage::IndexType inputIndex;
			typename TInputImage::SizeType  inputLargSize;
			typename TInputImage::IndexType inputLargIndex;
			typename TOutputImage::SizeType  outputSize;
			typename TOutputImage::IndexType outputIndex;

			outputIndex = this->GetOutput()->GetRequestedRegion().GetIndex();
			outputSize = this->GetOutput()->GetRequestedRegion().GetSize();
			inputLargSize = this->GetInput()->GetLargestPossibleRegion().GetSize();
			inputLargIndex = this->GetInput()->GetLargestPossibleRegion().GetIndex();

			for(unsigned int i=0; i<TInputImage::ImageDimension; i++)
			{
				inputSize[i] = outputSize[i];
				inputIndex[i] = outputIndex[i];
			}

			RequestedRegion.SetSize(inputSize);
			RequestedRegion.SetIndex(inputIndex);
			InputImagePointer input = const_cast< TInputImage *> ( this->GetInput() );
			input->SetRequestedRegion (RequestedRegion);
		}


		itkDebugMacro("GenerateInputRequestedRegion End");
	}


	/**
	* GenerateData Performs the accumulation
	*/
	template <class TInputImage, class TOutputImage>
	void PhaseSymmetryImageFilter<TInputImage,TOutputImage>::GenerateOutputInformation( void )
	{
		typename TOutputImage::RegionType outputRegion;
		typename TInputImage::IndexType inputIndex;
		typename TInputImage::SizeType  inputSize;
		typename TOutputImage::SizeType  outputSize;
		typename TOutputImage::IndexType outputIndex;
		typename TInputImage::SpacingType inSpacing;
		typename TInputImage::PointType inOrigin;
		typename TOutputImage::SpacingType outSpacing;
		typename TOutputImage::PointType outOrigin;

		// Get pointers to the input and output
		typename Superclass::OutputImagePointer output = this->GetOutput();
		typename Superclass::InputImagePointer input = const_cast< TInputImage *>( this->GetInput() );

		//Return if input and output are both null
		if( !input || !output )
		{
			return;
		}

		inputIndex = input->GetLargestPossibleRegion().GetIndex();
		inputSize = input->GetLargestPossibleRegion().GetSize();
		inSpacing = input->GetSpacing();
		inOrigin = input->GetOrigin();

		// Set the LargestPossibleRegion of the output.

		for(unsigned int i = 0; i<InputImageDimension; i++)
		{
			outputSize[i]  = inputSize[i];
			outputIndex[i] = inputIndex[i];
			outSpacing[i] = inSpacing[i];
			outOrigin[i]  = inOrigin[i];
		}

		//Set the size of the output region
		outputRegion.SetSize(outputSize);
		//Set the index of the output region
		outputRegion.SetIndex(outputIndex);
		//Set the origin and spacing
		output->SetOrigin(outOrigin);
		output->SetSpacing(outSpacing);
		//Set the largest po
		output->SetLargestPossibleRegion(outputRegion);
	}


	template <class TInputImage, class TOutputImage>
	void PhaseSymmetryImageFilter<TInputImage,TOutputImage>::PrintSelf(std::ostream& os, Indent indent) const
	{
		Superclass::PrintSelf(os,indent);

		//  os << indent << " Integral Filter Normalize By: " << m_Cutoff << std::endl;

	}



} // end namespace itk




#endif
